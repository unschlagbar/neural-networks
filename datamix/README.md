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
```

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
cache = "target/datamix-llm-cache"      # empty disables caching
api_key = ""                            # only if your server wants one
```

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

**Caching.** Every completion is keyed on model, temperature, prompts and call
index, and stored under `cache`. Rebuilding a mixture re-reads the cache and
makes no calls; raising `count` only pays for the new calls. Delete the
directory to regenerate from scratch.

## Report

`build` and `check` both print (and `build` writes) a per-source table: how much
was read, how much survived, how many tokens went into the mixture, the achieved
share against the requested one, and how many epochs each source actually ran.
Underneath it, every filter rejection is broken out by reason — the fastest way
to find a gate that is quietly eating a corpus.
