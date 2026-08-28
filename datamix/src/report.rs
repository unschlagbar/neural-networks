// Build report: what each source contributed, and what the filters removed.

use crate::config::Mix;
use crate::mix::{BuildStats, human, unit_name};

pub fn render(mix: &Mix, st: &BuildStats) -> String {
    let mut s = String::new();
    s.push_str("# datamix report\n\n");
    s.push_str(&format!(
        "output kind `{}` -> `{}`\n\n",
        mix.output.kind, mix.output.path
    ));
    s.push_str(&format!(
        "- records written: **{}** ({} tokens)\n",
        st.written,
        human(st.written_tokens)
    ));
    if st.held_out > 0 {
        s.push_str(&format!("- held out for eval: {}\n", st.held_out));
    }
    if st.dropped_shape > 0 {
        s.push_str(&format!(
            "- dropped (wrong shape for this output kind): {}\n",
            st.dropped_shape
        ));
    }
    s.push_str(&format!(
        "- budget: {} {}\n",
        human(st.budget),
        unit_name(st.unit)
    ));
    // What limited the size. With an explicit budget this is the only place it
    // is said: the corpus is simply smaller than asked for.
    let short: Vec<_> = st.sources.iter().filter(|s| s.is_short()).collect();
    if short.is_empty() {
        s.push_str("- every source filled its share\n\n");
    } else {
        let missing: usize = short
            .iter()
            .map(|s| s.wanted - s.emitted_units())
            .sum();
        s.push_str(&format!(
            "- **{} short of the budget**: {} ran out of data (asked for {}, could give {}) \
             — raise `epochs`, lower `weight`, or add data\n\n",
            human(missing),
            short
                .iter()
                .map(|s| format!("`{}`", s.name))
                .collect::<Vec<_>>()
                .join(", "),
            human(short.iter().map(|s| s.wanted).sum::<usize>()),
            human(short.iter().map(|s| s.emitted_units()).sum::<usize>()),
        ));
    }

    s.push_str("## sources\n\n");
    s.push_str("| source | read | kept | kept tok | in mix | mix tok | share | epochs | capped |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    let total: usize = st.sources.iter().map(|s| s.emitted_units()).sum();
    for src in &st.sources {
        let actual = if total > 0 {
            100.0 * src.emitted_units() as f32 / total as f32
        } else {
            0.0
        };
        let capped = if src.is_short() {
            format!("yes, wanted {}", human(src.wanted))
        } else {
            String::from("no")
        };
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.1}% (want {:.1}%) | {:.2} | {capped} |\n",
            src.name,
            human(src.read),
            human(src.kept),
            human(src.kept_tokens),
            human(src.emitted),
            human(src.emitted_tokens),
            actual,
            100.0 * src.share,
            src.epochs_used,
        ));
    }

    s.push_str("\n## filter rejections\n\n");
    for src in &st.sources {
        if src.rejects.is_empty() {
            continue;
        }
        let dropped: usize = src.rejects.values().sum();
        s.push_str(&format!(
            "- **{}**: {} of {} dropped ({:.1}%)\n",
            src.name,
            human(dropped),
            human(src.read),
            if src.read > 0 {
                100.0 * dropped as f32 / src.read as f32
            } else {
                0.0
            }
        ));
        let mut reasons: Vec<_> = src.rejects.iter().collect();
        reasons.sort_by_key(|&(_, &n)| std::cmp::Reverse(n));
        for (r, n) in reasons {
            s.push_str(&format!("  - {}: {}\n", r.label(), human(*n)));
        }
    }
    if !st.out_paths.is_empty() {
        s.push_str("\n## files\n\n");
        for p in &st.out_paths {
            let size = std::fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0);
            s.push_str(&format!("- `{p}` ({} B)\n", human(size)));
        }
    }
    s
}
