# datamix

Dataset mixing and filtering for the models in this repo. A *mixture file*
(`mixes/*.toml`) names the inputs, their weights and the quality gates; `datamix`
stages them, deduplicates, draws the mixture and writes either an SFT JSONL
(what `hqg` / `src/sft.rs` reads) or a `<|endoftext|>`-separated pretraining
corpus (what `ChunkedWordDataSet` reads). Nothing is compiled in — changing the
corpus is editing a text file, and the schema next door
(`mixture.schema.json`, named by the `#:schema` line at the top of each
mixture) makes the editor complete the keys and flag the ones that do not
exist.

```bash
cargo build -p datamix --release

./target/release/datamix check  mixes/pretrain.toml        # stats only, writes nothing
./target/release/datamix sample mixes/assistant_sft.toml -n 8
./target/release/datamix build  mixes/assistant_sft.toml   # write corpus + report
./target/release/datamix verify data/mix/assistant.jsonl # load it as training does
./target/release/datamix synth  mixes/synth/apps.syn -n 10
./target/release/datamix ping   mixes/assistant_llm.toml  # check the local LM server
./target/release/datamix edit   data/sources/handmade.jsonl  # write records by hand
```

## Writing records by hand

The same file has two front ends: `datamix edit` below, and the **corpus page**
in `harness` (the CORPUS chip in its status rail, or `HARNESS_PAGE=corpus`),
which edits `data/sources/handmade.jsonl` directly and masks by selecting text in
the turn itself.


`datamix edit [file.jsonl] [--port 7878]` serves a small editor on
`127.0.0.1` and reads and writes that file directly — nothing is downloaded,
nothing is copied back by hand. It is the way to add the handful of examples no
corpus and no template covers; point a `kind = "jsonl"` source at the file and
it mixes in like any other.

```toml
[source.handmade]
kind = "jsonl"
path = "data/sources/handmade.jsonl"
weight = 1
```

A record is either a conversation of role-tagged turns (`user`, `assistant`,
`tool`, `system`) or a single instruction/context/response pair. Text is typed
as text: the page escapes it into JSON itself, so a newline or a quote inside a
turn needs no thought. Under the turns it renders the **wire form** — the same
assembly `build_conversation` in `src/sft.rs` does, with `<CONTEXT>`, `<SEP>`,
`<result>…</result>` and `<END>` where they will land, and whitespace shown as
`·` / `¶` on request — and highlights the spans that carry loss.

The loss mask is the **"train on this"** checkbox on each assistant turn. On,
the turn is an `assistant` turn and its gradient trains the model. Off, it is
written as `assistant_context`: laid out identically, `<END>` included, read by
the backbone, and never a target. That is how a conversation can contain a
wrong tool call the model then recovers from without the mistake being trained.

Part of an answer can be masked the same way: select it and press **don't train
selection**. The selection snaps out to word borders — the unit the mask is
really applied in — and is written as a `no_loss` range on the message:

```json
{"role": "assistant", "content": "hmm, five. no wait, four.", "no_loss": [[0, 11]]}
```

The whole answer is still written and read; only the unmasked part carries
gradient, which the wire form shows directly (`hmm, five. ` plain, `no wait,
four.<END>` green). Masked ranges show up as struck-through chips under the
turn — click one to train it again — and they follow the text when it is
edited, so a mask does not have to be redrawn after a typo fix.

Saving writes through a temp file and then loads the result with the training
loader (`sft::load_jsonl`), so the status line is what `hqg` will actually see —
including a record it drops and why.

## How a build runs

1. **Stage.** Every source streams through its filters into a shard under
   `target/datamix-stage`. Nothing holds a corpus in RAM; the shard gives the
   mixer random access and the exact token count of what survived.
2. **Size the shares.** `weight` is a *share of the output*, relative to
   the other weights. With no `tokens` budget the corpus grows until the first
   source would have to repeat more than its `epochs` allows — that source is
   named in the output, because it sets the size of everything.
3. **Draw.** Each source's records are shuffled and taken until it has covered
   its share; a source with `epochs > 1` is cycled (reshuffled per pass). A
   source that runs out before filling its share is named on stdout and in the
   report (`capped` column), with what it asked for and what it could give —
   otherwise a corpus that came out half the size you asked for tells you
   nothing about why.
4. **Write.** Shuffled across sources, optionally split into shards and with a
   `holdout` fraction diverted to `<path>.eval`. The report lands next to it.

`weight_by = "records"` counts one record as one, whatever its length. On an SFT
mix that is the difference between a corpus and a caricature of it: a tool call
is a handful of tokens and a SmolTalk conversation is thousands, so a source
weighted at 12% of the *tokens* supplies most of the *questions*.

Everything is seeded from `[output] seed`: the same mixture file always builds
the same corpus.

## Mixture file

TOML, so any editor's TOML support checks the syntax as you type, and the
schema checks the keys.

```toml
#:schema ../datamix/mixture.schema.json

[output]
kind = "sft"             # "sft" | "text" | "none" (report only)
path = "data/mix/assistant.jsonl"
report = "data/mix/assistant_report.md"
seed = 1234
weight_by = "tokens"     # "tokens" | "records" — what a weight is a share of
tokens = 0               # 0 = as large as the weights allow (see step 2)
records = 0              # the budget when weight_by = "records"
shuffle = true
holdout = 0.02           # fraction diverted to <path>.eval
shard_bytes = 0          # >0 splits a text corpus into out.000.txt, ...

[filter]                 # defaults; every key may be repeated inside a source
min_bytes = 200
max_bytes = "400k"       # k / M / G suffixes, or a plain integer
max_words = 4096         # THE cap that matters: the unit the backbone unrolls
                         # in, counted with the model's own splitter (wordseg)
max_tokens = 28672       # byte tokenizer: one UTF-8 byte is one token
min_words = 0
min_alpha_ratio = 0.4    # letters+digits share — kills base64 and binaries
max_line_bytes = 2000    # kills minified / generated files
max_dup_line_ratio = 0.35    # kills log dumps and repeated boilerplate
trim_turns = true        # chat: keep the exchanges that fit rather than drop
                         # the conversation (98% vs 56% of SmolTalk at 4096)
must_contain = ["fn", "impl"]        # keep only documents matching one of these
must_not_contain = "TODO: generated" # a lone string is a one-element list
dedup = "near"           # "off" | "exact" | "near" (MinHash over 5-grams)
languages = ["en", "de"] # parquet `language` column only

[source.rust-lib]
kind = "dir"             # file | dir | parquet | chat | dolly | jsonl | synth | llm
path = "data/rust-lib"
ext = ["rs"]             # dir only; empty = every file
weight = 3
epochs = 1               # how often it may repeat to reach its share
```

Long prompts are where the format earns its keep — a multi-line string with
`\` at the end of a line joins into one flowing paragraph:

```toml
prompt = """
Here is one example:
{seed}

Write {n} more, varying the phrasing hard. Keep the tool-call syntax \
identical and never invent a tool.
"""
```

`kind = "chat"` reads a parquet whose messages column is a
`list<struct<role, content>>` — the shape SmolTalk, UltraChat and friends ship
in — one conversation per row (`role_column` / `content_column` name the leaves
if they are called something else). A turn whose role is not
system/user/assistant/tool is dropped, and so is a conversation left without an
exchange.

`select_column` and `select_values` keep only the rows whose flat subset column
matches, so one corpus can be several weighted sources:

```toml
[source.smoltalk_short]
kind = "chat"
path = "data/smol-smoltalk"
select_column = "source"
select_values = ["everyday-conversations", "explore-instruct-rewrite"]
```

That is not a nicety on SmolTalk: its subsets differ by 10x in answer length
(123 bytes for `explore-instruct-rewrite`, 1451 for `smol-magpie-ultra-short`,
which is 59% of the rows), so taken as one blob the longest subset decides how
your model answers everything. Split, that length is a weight you set.

A `dir` path, or a `parquet` or `chat` path pointing at a directory, reads every
matching file in sorted order; `skip_files` and `max_files` cut that listing down, which
is how you resume a sharded corpus you have already trained on
(`skip_files = 6` starts at the seventh shard).

Per-kind keys: `separator` (`file`/`dir`, default `<|endoftext|>`, empty =
whole file is one document), `text_column` / `language_column` (parquet,
`text_column` doubles as the field name for `jsonl`), `count` (`synth`, 0 =
every combination), `category` (tag on SFT records that carry none).

An unknown key or table, and a value of the wrong type, are errors naming the
line — a filter that is silently ignored is a corpus you cannot trust. The
schema reports the same things in the editor, before you run anything.

## Synthetic sources

No public dataset knows your lamp, your apps, your tool names or your
assistant's own name, so those examples are generated from templates
(`mixes/synth/*.syn`):

```ini
list app = Firefox ; firefox | Spotify ; spotify | the terminal ; kitty

template
instruction = {who are you|what's your name}?
response    = I'm Jarvis. I look after this machine.
category    = persona
```

A template written with `user =` / `assistant =` lines (as many as you like,
optionally opening with `system =`) produces a **conversation** instead. A
`tool =` line is what the call returned: it is prompt-side, so what the model
is trained to write is the reply *after* it. The lists bind once per example,
so a later turn can refer back to what an earlier one named — see
`mixes/synth/apps.syn`:

```ini
template
user      = {open|start} {app.0}
assistant = <tool>app.launch(name="{app.1}")</tool>
tool      = already_running
assistant = It's already open.
```

Two templates whose calls are identical and whose *results* differ are two
different examples, and that is the point: "already on" is only sayable once a
tool has said so. Give the same call every result it can return and the model
learns to read the result rather than to guess it.

An `assistant_context =` line is an assistant turn the model READS but is never
trained to produce, which is how a template teaches recovery from a mistake
without teaching the mistake:

```ini
template
user              = turn the light on
assistant_context = <tool>lamp.set(on=false)</tool>
tool              = already_off
assistant         = <tool>lamp.set(on=true)</tool>
tool              = on
assistant         = Sorry, I had that backwards. Light's on.
```

Only the last two assistant turns carry loss. Watch for the wrong turn and the
right one rendering to the *same* string — bind them to one list entry (see
`mixup` in `apps.syn`), or the "mistake" is trained after all.

A list entry's `;`-separated fields keep the spoken form and the identifier
bound to the same entry: `{app.0}` is what a person says, `{app.1}` what the
tool call needs. `{a|b|c}` is an inline paraphrase drawn per example — that is
where phrasing variety comes from, and it is *not* trimmed, so `{|s}` and
`{%| percent}` are how you write an optional word. `count` draws that many
examples (deduplicated on the rendered instruction); `count = 0` emits the full
cartesian product.

The tool-call syntax in the response is a convention of *your* data — the model
only learns to emit the string. Whatever parses it on the other side has to
agree with these files.

## What the two output kinds look like

**`kind = "sft"`** — JSONL, one record per line, in either of the two shapes
`src/sft.rs` reads. A single exchange:

```json
{"instruction": "open Firefox", "context": "", "response": "<tool>app.launch(name=\"firefox\")</tool>", "category": "apps"}
```

or a conversation of any length:

```json
{"messages": [{"role": "user", "content": "turn on the light"},
              {"role": "assistant", "content": "<tool>lamp.set(on=true)</tool>"},
              {"role": "tool", "content": "already_on"},
              {"role": "assistant", "content": "It's already on."}],
 "category": "lamp"}
```

A record is capped in **words**, not bytes: `max_words` counts through the same
`wordseg` splitter and the same `sft::build_example_turns` the trainer uses, so
a record that passes the filter is a record the trainer accepts. `max_tokens`
sits behind it as a guard against pathological all-long-word records. (Getting
this wrong is expensive: a 4096-*token* cap keeps 35% of SmolTalk, a 4096-*word*
cap keeps 99.3%.)

Both end up as one training window: `user <SEP> assistant <END>` repeated, with
**every assistant turn masked into the loss and every user turn out of it**. A
`system` message takes the `<CONTEXT>` slot of the first turn, so multi-turn
data needs no new tokens and no checkpoint surgery. `datamix verify` reports how
many records were multi-turn and how many exchanges the corpus holds.

**`kind = "text"`** — the documents concatenated with `<|endoftext|>` between
them, which is exactly what `ChunkedWordDataSet` already reads: it splits on the
separator and tokenizes each document on its own, so **the separator is never a
token and no end-of-document symbol is trained**. Same scheme as the existing
corpus, nothing new to learn. With `shard_bytes > 0` the file is split at
document borders into `out.000.txt`, `out.001.txt`, …

`<path>.eval` (from `holdout`) is byte-for-byte the same format, sliced off the
shuffled mixture rather than its tail.

## The local model (LM Studio and friends)

Two jobs a local OpenAI-compatible server does better than a template: writing
examples in phrasings you would not think of, and vetting what came out. Both
are opt-in — a mixture with neither key never opens a socket.

```toml
[llm]
endpoint = "http://localhost:1234/v1"   # LM Studio's default; http only
model = ""                              # empty = whatever the server loaded
temperature = 1.0
max_tokens = 1024
timeout = 300                           # seconds for one completion
retries = 2
cache = "data/llm-cache"      # empty disables caching
api_key = ""                            # only if your server wants one
workers = 4                             # requests in flight; match the server's PARALLEL
```

**Two jobs a local server does badly**, and the one knob that fixes both: a
reasoning model spends most of every completion thinking. On Python→Rust
rewrites of `assistant_qa` that was 60–90% of the budget, and two replies in
four ran out of tokens mid-answer. None of `chat_template_kwargs`,
`/no_think` or `reasoning.enabled` turns it off — LM Studio ignores all three.
What works is taking the template over:

```toml
completion_stop = "<|im_end|>"
completion_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
```

A non-empty `completion_template` switches every call to `/v1/completions`,
where the prompt is passed verbatim — the only way to prefill the assistant
turn. The model reads its own `<think></think>` as already closed and answers
directly: **4.3 s per record against 43.6 s**, with no measured quality loss.
The template is part of the cache key, so a deliberated answer and a prefilled
one never share an entry. Leave it empty for a non-reasoning model.

`workers` is how many requests are in flight. Set it to the server's parallel
slot count (`lms ps` prints it as PARALLEL); past that the aggregate rate does
not move — measured 99 tok/s at both 4 and 8.

`datamix ping [mix.toml]` lists the server's loaded models and runs one
completion — do that before a long build.

**Generating** (`kind = llm`):

```toml
[source.lamp_llm]
kind = "llm"
count = 600                              # examples wanted
batch = 10                               # examples asked for per call
seed_file = "mixes/synth/lamp.syn"       # rendered into {seed}
seeds = 4                                # how many examples per call
model = ""                               # optional per-source override
temperature = 1.2                        # optional; < 0 inherits [llm]
system = "You answer ONLY with JSON lines, one object per line, keys: instruction, context, response."
prompt = """
Here is one example:
{seed}

Write {n} more, varying the phrasing hard.
"""
```

The reply is read as JSONL — one `{instruction, context, response}` object per
line, or one `{"messages": [...]}` conversation per line if that is what you
asked for — so a line the model fumbles costs one example instead of the whole
call;
code fences and a `<think>` prelude are stripped. `{seed}` is what keeps a
generator honest: each call carries `seeds` real examples **rendered in the
exact output format**, so the model varies the phrasing instead of inventing
its own tool syntax. That detail is not cosmetic — seeding with instructions
alone produced `turn_lights_off(room='dining room')` where the corpus uses
`<tool>lamp.set(on=true)</tool>`. Generation stops early if
five calls in a row parse to nothing.

**Filtering by prompt** (`judge`, in `[filter]` or in one source):

```toml
judge = "Keep an example only if the response contains a well-formed tool call that matches the request."
judge_expect = "yes"       # the reply must start with this word
judge_model = ""           # empty inherits [llm] model
judge_temperature = 0.0    # judging is a classification, not a generation
```

**Judge at temperature 0** — it defaults there for a reason. The same judge
inheriting a generator's `temperature = 1.0` waved through an invented
`lamp.dim(...)` call with a duplicated `</tool>`; at 0 it rejected both. Give it
the tool contract explicitly (the exact tool names, argument names and allowed
values); a vague "keep the good ones" catches far less.

The judge runs *after* the cheap gates, on what survived them, and costs one
round trip per record. Rejections appear in the report as `rejected by the
judge`, like every other filter.

`kind = llm` sources generate in rounds of `[llm] workers`, the same pool the
rewrite uses: one call is ~100 s of a 27B writing a batch of conversations, and
one at a time leaves the rest of the server's slots idle — measured 2.8
records/min serial against 16 in rounds of eight. The seeds are drawn per call,
so the calls of one round do not all ask the same thing. What follows a round —
the filters, the judge, the shard write — stays serial and in order.

**Rewriting what a source already holds** (`transform`, in `[filter]` or in one
source):

```toml
transform_when = ["```python", "```py", "python", "Python"]   # empty = every record
transform_temperature = 0.2                                   # a translation, not a generation
transform = """
You convert Python programming data into Rust. You are given one conversation
as {"messages": [...]}. Rewrite it so no trace of Python is left … Reply with
ONLY the rewritten {"messages": [...]} JSON object.
"""
```

Where `judge` decides whether a record survives, `transform` decides what it
says. The record goes up in the same `{"messages": [...]}` shape the SFT loader
reads, and the reply must come back with **the same turns in the same order with
the same roles** — a reply that drops a turn or relabels one is refused and the
record is dropped, counted in the report as `rewrite failed`. That check is not
pedantry: an assistant turn relabelled as a user turn would silently move the
SFT loss mask onto the prompt.

A dolly-shaped record travels as the turns it stands for (`context` as the
system turn), so one prompt handles both shapes.

The rewrite is keyed on the record's own text, which is what makes it stick: the
first build pays for it once and every later build reads the answers back
without opening a socket — measured 66 s, then 0.005 s for byte-identical
output. `transform_when` is what keeps a whole-corpus rewrite affordable; a
record that matches none of its substrings never reaches the server. On
`mixes/assistant_qa.toml` that is 24% of the records, ~52k of them, about 56
hours on a 27B at four workers — a number worth knowing before you start, and
resumable at any point because nothing is ever asked twice.

**Making a rewrite into a corpus** (`transform_only`). A rewrite that expensive
should not live only inside whatever corpus was built last. `transform_only =
true` keeps ONLY the records the rewrite applied to, dropping everything
`transform_when` did not match (counted as `outside the rewrite`), so a mixture
can write the translations out as a corpus of their own. With `cache_only =
true` beside it, that mixture opens no socket at all — it assembles exactly what
has already been paid for. `mixes/rustified.toml` is that mixture:

```
cargo run -p datamix --release -- build mixes/rustified.toml
  -> data/sources/rustified.jsonl        3,141 records, ~4.4M tokens
```

and every other mixture then reads the translations with three lines and no
rewrite prompt of its own:

```toml
[source.rustified]
kind = "dolly"          # `dolly`, not `jsonl`: it reads both record shapes
path = "data/sources/rustified.jsonl"
```

The mixer balances *between* sources, so list every subset under ONE source
here — splitting them meant the smallest yield capped the rest, and 3,141
translated records came out as 286.

The cache itself lives in `data/llm-cache`, deliberately outside `target/`: what
accumulates there is the corpus, not a build artifact, and `cargo clean` used to
take 56 hours of it.

**Compiling what the rewrite produced** (`verify`, `verify_repair`):

```toml
verify = "rust"            # extract the ```rust blocks and run rustc; "" = off
verify_repair = """
You are fixing Rust code inside one training conversation... Fix only what the
errors point at. Reply with ONLY the corrected {"messages": [...]} JSON object.
"""
```

A model translating code makes the kind of mistake that reads fine and does not
build — a string literal eaten by a substitution (`Node { name: ".to_string()" }`),
a crate invented to stand in for a Python library. `rustc` settles that exactly,
in ~0.3 s against the ~4 s the rewrite itself cost. A snippet is tried as a file
of items and then wrapped in a function, because instruction data shows both a
full `fn` and a bare statement; only the second needs the wrapper. No `rustc` on
PATH skips the check rather than rejecting everything.

When it fails and `verify_repair` is set, the record goes back up with rustc's
own diagnostics attached — the compiler describes the defect better than any
rule could — and is compiled again. Measured on 24 rewritten records: 63% built
as they came back, one repair pass took that to 79%, and a second pass fixed
nothing, which is why there is only one. What still fails is dropped, counted as
`code does not compile`.

**Cutting the boilerplate out of system turns** (`system_strip`, `system_fold`):

```toml
system_strip = ["You're an AI assistant for text re-writing.", "You are an AI assistant."]
system_fold = false
```

A corpus's system turns are rarely as varied as they look: on `assistant_qa`
they are 246 distinct strings of which twelve cover 32% of every record, and one
identity sentence opens 13.6% of them. That is a prop — the model learns to
answer with it in front of the prompt and is off its distribution the moment it
is not there. `system_strip` cuts those sentences out; a turn that was *only*
framing disappears.

What it deliberately does not do is delete the turn. In ~23k records the system
turn is not framing but the only instruction the record has ("Rewrite the input
text to make it more professional", the user turn holding just the raw email);
cutting it whole would leave a corpus of unanswerable examples. `system_fold =
true` goes one step further and moves the surviving instruction into the first
user turn, which is where inference puts it.

Both run before the rewrite, because the record's own text is its cache key: a
system turn edited afterwards would hash two ways and re-translate.

**Caching.** Every completion is keyed on model, temperature, template, prompts
and call index, and stored under `cache`. Rebuilding a mixture re-reads the cache and
makes no calls; raising `count` only pays for the new calls. Delete the
directory to regenerate from scratch.

## Report

`build` and `check` both print (and `build` writes) a per-source table: how much
was read, how much survived, how many tokens went into the mixture, the achieved
share against the requested one, and how many epochs each source actually ran.
Underneath it, every filter rejection is broken out by reason — the fastest way
to find a gate that is quietly eating a corpus.
