// Template expansion: instruction data for skills no public corpus covers
// (your apps, your lamp, your tool names, your assistant's own identity).
//
// A `.syn` file is lists plus templates:
//
//   list app   = Firefox ; firefox | Spotify ; spotify
//   list state = on ; true | off ; false
//
//   template
//   instruction = {open|start} {app.0}
//   response    = <tool>app.launch(name="{app.1}")</tool>
//   category    = apps
//
// A list entry has `;`-separated fields: `{app.0}` is what a person says,
// `{app.1}` what the tool call needs, so the surface form and the identifier
// stay bound to the same entry. `{a|b|c}` is an inline paraphrase choice, drawn
// per example — that is where the phrasing variety comes from.
//
// One example is: pick one entry per referenced list, resolve every inline
// choice, render each field. `count` on the source decides how many are drawn;
// with `count = 0` the full cartesian product is emitted.
//
// A template that uses `user =` / `assistant =` lines instead of
// instruction/response is a **conversation**, and may have as many turns as it
// likes — the lists bind once per example, so a later turn can refer back to
// the same app the first one used. A `tool =` line is what the call returned:
// it is prompt-side, so the reply after it is what the model is trained to
// write.
//
//   template
//   user      = {open|start} {app.0}
//   assistant = <tool>app.launch(name="{app.1}")</tool>
//   tool      = ok
//   assistant = {app.0} is up.

use std::collections::HashMap;

use neural_networks::sft::{Role, Turn};

use crate::config::Result;
use crate::record::Record;
use crate::rng::Rng;

struct Template {
    instruction: String,
    context: String,
    response: String,
    category: String,
    /// `user`/`assistant`/`system`/`tool` lines, in the order written. Non-empty
    /// makes this a conversation template and the single-turn fields unused.
    turns: Vec<(Role, String)>,
    /// Lists this template references, in a fixed order (the odometer axes).
    lists: Vec<String>,
}

pub struct Synth {
    lists: HashMap<String, Vec<Vec<String>>>,
    templates: Vec<Template>,
}

/// Hard cap on a full-cartesian expansion, so a typo in a list cannot ask for
/// a billion examples.
const MAX_FULL: usize = 500_000;

pub fn load(path: &str) -> Result<Synth> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse(&text, path)
}

pub fn parse(text: &str, path: &str) -> Result<Synth> {
    let mut lists: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    let mut templates: Vec<Template> = Vec::new();
    let mut cur: Option<Template> = None;

    for (no, raw) in text.lines().enumerate() {
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        }
        .trim();
        if line.is_empty() {
            continue;
        }
        let at = |m: &str| format!("{path}:{}: {m}", no + 1);

        if let Some(rest) = line.strip_prefix("list ") {
            let (name, values) = rest
                .split_once('=')
                .ok_or_else(|| at("expected `list name = a | b | c`"))?;
            let entries: Vec<Vec<String>> = values
                .split('|')
                .map(|e| e.split(';').map(|f| f.trim().to_string()).collect())
                .filter(|e: &Vec<String>| !e.iter().all(|f| f.is_empty()))
                .collect();
            if entries.is_empty() {
                return Err(at("list has no entries"));
            }
            lists.insert(name.trim().to_string(), entries);
            continue;
        }

        if line == "template" {
            if let Some(t) = cur.take() {
                templates.push(t);
            }
            cur = Some(Template {
                instruction: String::new(),
                context: String::new(),
                response: String::new(),
                category: String::new(),
                turns: Vec::new(),
                lists: Vec::new(),
            });
            continue;
        }

        let t = cur
            .as_mut()
            .ok_or_else(|| at("key outside a `template` block"))?;
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| at("expected `key = value`"))?;
        let v = unescape(v.trim());
        match k.trim() {
            "instruction" => t.instruction = v,
            "context" => t.context = v,
            "response" => t.response = v,
            "category" => t.category = v,
            "user" => t.turns.push((Role::User, v)),
            "assistant" => t.turns.push((Role::Assistant, v)),
            "system" => t.turns.push((Role::System, v)),
            "tool" => t.turns.push((Role::Tool, v)),
            // An assistant turn the model reads but is never trained to write:
            // the mistake in an error-recovery conversation.
            "assistant_context" | "assistant_noloss" => {
                t.turns.push((Role::AssistantContext, v))
            }
            other => return Err(at(&format!("unknown template key '{other}'"))),
        }
    }
    if let Some(t) = cur.take() {
        templates.push(t);
    }
    if templates.is_empty() {
        return Err(format!("{path}: no `template` blocks"));
    }

    for t in &mut templates {
        let mut names = Vec::new();
        let turn_texts: Vec<&String> = t.turns.iter().map(|(_, v)| v).collect();
        for field in [&t.instruction, &t.context, &t.response]
            .into_iter()
            .chain(turn_texts)
        {
            for r in refs(field) {
                if !lists.contains_key(&r) {
                    return Err(format!("{path}: template references unknown list '{r}'"));
                }
                if !names.contains(&r) {
                    names.push(r);
                }
            }
        }
        t.lists = names;
        if t.turns.is_empty() && (t.instruction.is_empty() || t.response.is_empty()) {
            return Err(format!(
                "{path}: a template needs either instruction+response or \
                 user/assistant lines"
            ));
        }
        if !t.turns.is_empty() && !t.turns.iter().any(|(r, _)| *r == Role::Assistant) {
            return Err(format!("{path}: a conversation template has no assistant turn"));
        }
    }
    Ok(Synth { lists, templates })
}

impl Synth {
    /// Every combination of every template, as an upper bound on `count`.
    pub fn combinations(&self) -> usize {
        self.templates
            .iter()
            .map(|t| {
                t.lists
                    .iter()
                    .map(|n| self.lists[n].len())
                    .product::<usize>()
            })
            .sum()
    }

    /// Expand into records. `count = 0` emits the full cartesian product of
    /// every template (capped); otherwise `count` examples are drawn, split
    /// evenly across templates, and de-duplicated on the rendered instruction.
    pub fn expand(&self, count: usize, rng: &mut Rng, default_category: &str) -> Vec<Record> {
        let mut out = Vec::new();
        if count == 0 {
            for t in &self.templates {
                self.enumerate(t, &mut out, rng, default_category);
            }
            return out;
        }
        let per = count.div_ceil(self.templates.len());
        let mut seen = std::collections::HashSet::new();
        for t in &self.templates {
            let mut made = 0usize;
            // Enough tries to fill even a template whose combinations barely
            // exceed its quota, without spinning on an exhausted one.
            let mut tries = 0usize;
            while made < per && tries < per * 8 + 64 {
                tries += 1;
                let pick: Vec<&Vec<String>> = t
                    .lists
                    .iter()
                    .map(|n| rng.pick(&self.lists[n]))
                    .collect();
                let rec = self.render(t, &pick, rng, default_category);
                // The whole rendered record is the key. Keying on the prompt
                // alone throws away every variation a template puts in its
                // answer, which for a template whose point IS the answer — the
                // same question asked of a different tool result — collapses it
                // to a handful of examples.
                let key = match &rec {
                    Record::Sft {
                        instruction,
                        context,
                        response,
                        ..
                    } => format!("{instruction}\u{1}{context}\u{1}{response}"),
                    Record::Doc { text } => text.clone(),
                    // Every turn, not just the opener: templates that share an
                    // opening line on purpose — small talk before a tool call —
                    // are different examples from the second turn onwards.
                    Record::Chat { turns, .. } => {
                        turns.iter().map(|t| t.content.as_str()).collect::<Vec<_>>().join("\u{1}")
                    }
                };
                if seen.insert(key) {
                    out.push(rec);
                    made += 1;
                }
            }
        }
        out
    }

    /// Odometer over the template's lists — every combination exactly once.
    fn enumerate(&self, t: &Template, out: &mut Vec<Record>, rng: &mut Rng, cat: &str) {
        let axes: Vec<&Vec<Vec<String>>> = t.lists.iter().map(|n| &self.lists[n]).collect();
        let total: usize = axes.iter().map(|a| a.len()).product();
        let mut idx = vec![0usize; axes.len()];
        for _ in 0..total.min(MAX_FULL) {
            let pick: Vec<&Vec<String>> =
                axes.iter().zip(&idx).map(|(a, &i)| &a[i]).collect();
            out.push(self.render(t, &pick, rng, cat));
            for (d, a) in idx.iter_mut().zip(&axes) {
                *d += 1;
                if *d < a.len() {
                    break;
                }
                *d = 0;
            }
        }
    }

    fn render(&self, t: &Template, pick: &[&Vec<String>], rng: &mut Rng, cat: &str) -> Record {
        let bind: HashMap<&str, &Vec<String>> = t
            .lists
            .iter()
            .map(|n| n.as_str())
            .zip(pick.iter().copied())
            .collect();
        let category = if t.category.is_empty() {
            cat.to_string()
        } else {
            t.category.clone()
        };
        if !t.turns.is_empty() {
            return Record::Chat {
                turns: t
                    .turns
                    .iter()
                    .map(|(role, text)| Turn {
                        role: *role,
                        content: fill(text, &bind, rng),
                    })
                    .collect(),
                category,
            };
        }
        Record::Sft {
            instruction: fill(&t.instruction, &bind, rng),
            context: fill(&t.context, &bind, rng),
            response: fill(&t.response, &bind, rng),
            category,
        }
    }
}

/// List names referenced by `{name}` / `{name.field}` in one field.
fn refs(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for group in groups(s) {
        let choices = alternatives(&group);
        if choices.len() > 1 {
            // A list referenced only from inside a choice still has to be bound,
            // or `fill` will find no entry for it.
            for c in choices {
                out.extend(refs(c));
            }
            continue;
        }
        let name = group.split('.').next().unwrap_or("").trim();
        if !name.is_empty() {
            out.push(name.to_string());
        }
    }
    out
}

/// Byte index of the `}` closing the `{` at `open`, honouring nesting.
fn closing_brace(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a group on the `|` at ITS depth only. A nested group's alternatives
/// belong to that group, not to this one; splitting on every `|` tears
/// `{i said {a|b}|no}` into pieces that reassemble into neither.
fn alternatives(inner: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            '|' if depth == 0 => {
                out.push(&inner[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&inner[start..]);
    out
}

/// The contents of every top-level `{...}` group.
fn groups(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '{' => {
                depth += 1;
                if depth == 1 {
                    cur.clear();
                    continue;
                }
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                    continue;
                }
            }
            _ => {}
        }
        if depth > 0 {
            cur.push(c);
        }
    }
    out
}

fn fill(s: &str, bind: &HashMap<&str, &Vec<String>>, rng: &mut Rng) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let Some(close) = closing_brace(rest, open) else {
            out.push_str(&rest[open..]);
            return out;
        };
        let inner = &rest[open + 1..close];
        rest = &rest[close + 1..];
        // Not trimmed: a leading or trailing space inside a choice is how a
        // template writes an optional word (`{|s}`, `{%| percent}`).
        let choices = alternatives(inner);
        if choices.len() > 1 {
            // The chosen alternative may hold groups of its own — a list
            // reference inside a paraphrase is the common case.
            let choice = *rng.pick(&choices);
            out.push_str(&fill(choice, bind, rng));
            continue;
        }
        let (name, field) = match inner.split_once('.') {
            Some((n, f)) => (n.trim(), f.trim().parse::<usize>().unwrap_or(0)),
            None => (inner.trim(), 0),
        };
        match bind.get(name) {
            // A missing field falls back to field 0, so a one-field list can be
            // referenced as `{room.1}` in a template shared with richer lists.
            Some(entry) => out.push_str(entry.get(field).unwrap_or(&entry[0])),
            None => {
                out.push('{');
                out.push_str(inner);
                out.push('}');
            }
        }
    }
    out.push_str(rest);
    out
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYN: &str = "\
list room = kitchen ; kitchen | living room ; living_room
list state = on ; true | off ; false

template
instruction = {turn|switch} {state.0} the {room.0} light
response = <tool>lamp.set(room=\"{room.1}\", on={state.1})</tool>
category = smart_home
";

    #[test]
    fn fields_of_one_entry_stay_bound_together() {
        let syn = parse(SYN, "t.syn").unwrap();
        assert_eq!(syn.combinations(), 4);
        let mut rng = Rng::new(1);
        for rec in syn.expand(0, &mut rng, "x") {
            let Record::Sft {
                instruction,
                response,
                category,
                ..
            } = rec
            else {
                panic!("synth must produce SFT records")
            };
            assert_eq!(category, "smart_home");
            // The spoken room and the identifier come from the same entry.
            let spoken_living = instruction.contains("living room");
            assert_eq!(spoken_living, response.contains("living_room"));
            assert_eq!(instruction.contains(" on the"), response.contains("on=true"));
            assert!(!instruction.contains('{') && !response.contains('{'));
        }
    }

    #[test]
    fn full_expansion_covers_every_combination() {
        let syn = parse(SYN, "t.syn").unwrap();
        let mut rng = Rng::new(7);
        assert_eq!(syn.expand(0, &mut rng, "x").len(), 4);
    }

    #[test]
    fn inline_choices_keep_their_spacing() {
        let syn = parse(
            "list n = 10\n\ntemplate\ninstruction = set it to {n.0}{%| percent}\nresponse = ok\n",
            "t.syn",
        )
        .unwrap();
        let mut rng = Rng::new(3);
        for rec in syn.expand(0, &mut rng, "x") {
            let Record::Sft { instruction, .. } = rec else {
                unreachable!()
            };
            assert!(
                instruction.ends_with("10%") || instruction.ends_with("10 percent"),
                "{instruction}"
            );
        }
    }

    #[test]
    fn a_conversation_template_binds_one_entry_across_turns() {
        let syn = parse(
            "list room = kitchen ; kitchen | office ; office\n\n             template\n             user = turn on the {room.0} light\n             assistant = <tool>lamp.set(room=\"{room.1}\")</tool>\n             user = dimmer\n             assistant = <tool>lamp.brightness(room=\"{room.1}\")</tool>\n",
            "t.syn",
        )
        .unwrap();
        let mut rng = Rng::new(2);
        let recs = syn.expand(0, &mut rng, "x");
        assert_eq!(recs.len(), 2);
        for rec in recs {
            let Record::Chat { turns, .. } = rec else {
                panic!("user/assistant lines must make a conversation")
            };
            assert_eq!(turns.len(), 4);
            // The follow-up refers to the room the first turn named.
            let room = if turns[0].content.contains("kitchen") {
                "kitchen"
            } else {
                "office"
            };
            assert!(turns[3].content.contains(room), "{:?}", turns[3].content);
        }
    }

    #[test]
    fn a_conversation_template_needs_an_assistant_turn() {
        let err = parse("template\nuser = hello\n", "t.syn").err().unwrap();
        assert!(err.contains("no assistant turn"), "{err}");
    }

    /// A list reference inside a paraphrase choice. Matching the FIRST `}`
    /// instead of the closing one turns this into a list named
    /// `i said {mixup` and leaks the raw template into the corpus.
    #[test]
    fn a_choice_may_contain_a_list_reference() {
        let syn = parse(
            "list app = Firefox ; firefox | Discord ; discord\n\
             \n\
             template\n\
             instruction = {i said {app.0}|no, {app.0}|that one}\n\
             response = <tool>app.launch(name=\"{app.1}\")</tool>\n",
            "t.syn",
        )
        .unwrap();
        let mut rng = Rng::new(7);
        for rec in syn.expand(40, &mut rng, "t") {
            let Record::Sft { instruction, .. } = rec else {
                panic!("expected single-turn records")
            };
            assert!(
                !instruction.contains('{') && !instruction.contains('|'),
                "template syntax survived rendering: {instruction:?}"
            );
            assert!(
                instruction.contains("Firefox") || instruction.contains("Discord")
                    || instruction == "that one",
                "{instruction:?}"
            );
        }
    }

    /// The list is referenced ONLY from inside a choice, so `refs` has to see
    /// through the choice or nothing binds it.
    #[test]
    fn a_list_used_only_inside_a_choice_is_still_bound() {
        let syn = parse(
            "list app = Firefox ; firefox\n\
             \n\
             template\n\
             instruction = open {{app.0}|it}\n\
             response = done\n",
            "t.syn",
        )
        .unwrap();
        let mut rng = Rng::new(3);
        for rec in syn.expand(20, &mut rng, "t") {
            let Record::Sft { instruction, .. } = rec else { panic!() };
            assert!(
                instruction == "open Firefox" || instruction == "open it",
                "{instruction:?}"
            );
        }
    }

    #[test]
    fn unknown_list_is_an_error() {
        let err = parse("template\ninstruction = {nope}\nresponse = x\n", "t.syn")
            .err()
            .unwrap();
        assert!(err.contains("unknown list 'nope'"), "{err}");
    }
}
