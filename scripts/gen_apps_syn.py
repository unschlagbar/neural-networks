#!/usr/bin/env python3
"""Generate mixes/synth/apps.syn from a real pool of desktop application ids.

The file this writes replaces a hand-written one whose whole app universe was
seven lowercase ids (firefox, spotify, kitty, code, nautilus, discord, steam).
Measured on the corpus that produced: `app.list()` returned the SAME string 24
of 25 times, so the model learned it as a constant rather than as something to
read, and it never emitted an id shaped like `org.kde.dolphin` — while 33 of
this machine's 86 installed apps are shaped exactly that way.

Three things make the generated file generalize where the hand-written one
could not:

  * the installed list is drawn fresh for every entry, so `app.list()` never
    returns the same thing twice and reading it is the only way to answer;
  * ids carry the shapes a real desktop has - reverse-DNS, hyphenated,
    capitalized - not just plain lowercase words;
  * a spoken description ("the file manager") maps to a DIFFERENT id in
    different examples, which is what makes the list load-bearing instead of
    decorative.

    python3 scripts/gen_apps_syn.py [pool.json] [out.syn]
"""
import json, pathlib, random, sys

POOL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "datamix/pools/installed_apps.json")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "mixes/synth/apps.syn")
rng = random.Random(20260908)

# Category -> how a person refers to such a program. The point of this table is
# that several apps share one phrase: "the file manager" is dolphin here and
# thunar there, so the phrase alone can never determine the id.
ROLES = {
    "FileManager": ["the file manager", "the files app", "my file browser", "the file browser"],
    "WebBrowser": ["the browser", "my web browser", "a browser"],
    "TerminalEmulator": ["the terminal", "a terminal", "the console", "a shell"],
    "TextEditor": ["the text editor", "my editor", "a text editor"],
    "IDE": ["my code editor", "the IDE", "the code editor"],
    "Player": ["the media player", "the video player", "something to play this"],
    "AudioVideo": ["the media player", "the music player"],
    "RasterGraphics": ["the image editor", "the photo editor", "the paint program"],
    "Viewer": ["the image viewer", "the photo viewer"],
    "Archiving": ["the archive tool", "the zip tool"],
    "Calculator": ["the calculator"],
    "InstantMessaging": ["the chat app", "my messenger"],
    "Game": ["the game", "that game"],
    "Settings": ["the settings", "system settings", "the control panel"],
    "Monitor": ["the system monitor", "the task manager"],
    "Spreadsheet": ["the spreadsheet app", "the spreadsheet"],
    "WordProcessor": ["the word processor", "the writer"],
    "Presentation": ["the slides app", "the presentation app"],
    "Email": ["the mail client", "my email"],
    "Video": ["the video editor"],
    "Photography": ["the photo tool"],
    "3DGraphics": ["the 3d program"],
    "Development": ["the dev tool"],
}

# A general pool, so the corpus is not overfitted to exactly this machine: the
# ids a Linux desktop commonly carries, in the same four shapes.
GENERAL = [
    ("org.gnome.Nautilus", "Files", "FileManager"), ("thunar", "Thunar", "FileManager"),
    ("pcmanfm", "PCManFM", "FileManager"), ("nemo", "Nemo", "FileManager"),
    ("org.kde.dolphin", "Dolphin", "FileManager"),
    ("chromium", "Chromium", "WebBrowser"), ("brave-browser", "Brave", "WebBrowser"),
    ("vivaldi-stable", "Vivaldi", "WebBrowser"), ("librewolf", "LibreWolf", "WebBrowser"),
    ("org.mozilla.firefox", "Firefox", "WebBrowser"), ("com.brave.Browser", "Brave", "WebBrowser"),
    ("kitty", "kitty", "TerminalEmulator"), ("foot", "foot", "TerminalEmulator"),
    ("org.wezfurlong.wezterm", "WezTerm", "TerminalEmulator"),
    ("com.gexperts.Tilix", "Tilix", "TerminalEmulator"), ("st", "st", "TerminalEmulator"),
    ("gnome-terminal", "GNOME Terminal", "TerminalEmulator"),
    ("nvim", "Neovim", "TextEditor"), ("emacs", "Emacs", "TextEditor"),
    ("org.gnome.TextEditor", "Text Editor", "TextEditor"), ("sublime_text", "Sublime Text", "TextEditor"),
    ("org.gnome.gedit", "gedit", "TextEditor"), ("helix", "Helix", "TextEditor"),
    ("jetbrains-idea", "IntelliJ IDEA", "IDE"), ("jetbrains-clion", "CLion", "IDE"),
    ("codium", "VSCodium", "IDE"), ("org.kde.kdevelop", "KDevelop", "IDE"),
    ("zed", "Zed", "IDE"),
    ("vlc", "VLC", "Player"), ("io.mpv.Mpv", "mpv", "Player"),
    ("com.github.rafostar.Clapper", "Clapper", "Player"), ("celluloid", "Celluloid", "Player"),
    ("spotify", "Spotify", "AudioVideo"), ("org.strawberrymusicplayer.strawberry", "Strawberry", "AudioVideo"),
    ("rhythmbox", "Rhythmbox", "AudioVideo"), ("com.spotify.Client", "Spotify", "AudioVideo"),
    ("org.kde.elisa", "Elisa", "AudioVideo"),
    ("org.gimp.GIMP", "GIMP", "RasterGraphics"), ("krita", "Krita", "RasterGraphics"),
    ("org.inkscape.Inkscape", "Inkscape", "RasterGraphics"), ("pinta", "Pinta", "RasterGraphics"),
    ("org.darktable.Darktable", "darktable", "Photography"),
    ("org.gnome.Loupe", "Image Viewer", "Viewer"), ("feh", "feh", "Viewer"),
    ("org.kde.okular", "Okular", "Viewer"), ("org.gnome.Evince", "Document Viewer", "Viewer"),
    ("zathura", "Zathura", "Viewer"),
    ("org.gnome.FileRoller", "Archive Manager", "Archiving"), ("xarchiver", "Xarchiver", "Archiving"),
    ("galculator", "galculator", "Calculator"), ("org.gnome.Calculator", "Calculator", "Calculator"),
    ("discord", "Discord", "InstantMessaging"), ("im.riot.Riot", "Element", "InstantMessaging"),
    ("org.telegram.desktop", "Telegram", "InstantMessaging"),
    ("signal-desktop", "Signal", "InstantMessaging"), ("slack", "Slack", "InstantMessaging"),
    ("com.discordapp.Discord", "Discord", "InstantMessaging"),
    ("steam", "Steam", "Game"), ("net.lutris.Lutris", "Lutris", "Game"),
    ("com.heroicgameslauncher.hgl", "Heroic", "Game"), ("org.prismlauncher.PrismLauncher", "Prism Launcher", "Game"),
    ("htop", "htop", "Monitor"), ("org.gnome.SystemMonitor", "System Monitor", "Monitor"),
    ("org.kde.plasma-systemmonitor", "System Monitor", "Monitor"),
    ("thunderbird", "Thunderbird", "Email"), ("org.gnome.Evolution", "Evolution", "Email"),
    ("kdenlive", "Kdenlive", "Video"), ("org.shotcut.Shotcut", "Shotcut", "Video"),
    ("blender", "Blender", "3DGraphics"), ("freecad", "FreeCAD", "3DGraphics"),
    ("org.gnome.Settings", "Settings", "Settings"), ("xfce4-settings-manager", "Settings", "Settings"),
    ("transmission-gtk", "Transmission", "Development"), ("org.qbittorrent.qBittorrent", "qBittorrent", "Development"),
    ("virt-manager", "Virtual Machine Manager", "Development"), ("org.gnome.Boxes", "Boxes", "Development"),
    ("org.keepassxc.KeePassXC", "KeePassXC", "Development"), ("obsidian", "Obsidian", "TextEditor"),
]

def role_of(cats, name):
    for key in ROLES:
        if key.lower() in cats.lower():
            return key
    low = (cats + " " + name).lower()
    for key, _ in ROLES.items():
        if key.lower() in low:
            return key
    return None

apps = []          # (id, display name, role or None)
seen = set()
for a in json.loads(POOL.read_text()):
    if a["id"] in seen:
        continue
    seen.add(a["id"])
    apps.append((a["id"], a["name"], role_of(a["categories"], a["name"])))
for i, n, r in GENERAL:
    if i not in seen:
        seen.add(i)
        apps.append((i, n, r))

by_role = {}
for i, n, r in apps:
    if r:
        by_role.setdefault(r, []).append((i, n))

def installed(must=(), forbid=()):
    """A plausible `app.list()` answer: a random subset, always holding `must`.

    Sizes run from a handful to a few dozen. The long ones matter as much as the
    short: a real machine answers with 86 ids, and a model that has only ever
    read eight has no reason to keep reading at thirty.
    """
    n = rng.choice([6, 8, 9, 11, 12, 14, 16, 18, 22, 26, 31, 38, 45])
    pool = [i for i, _, _ in apps if i not in must and i not in forbid]
    picked = list(must) + rng.sample(pool, min(n, len(pool)))
    rng.shuffle(picked)
    return ", ".join(picked)

def naive(name):
    """The id a model would guess from a spoken name: lowercase, no spaces."""
    return "".join(ch for ch in name.lower() if ch.isalnum())

def esc(s):
    # `;` separates fields and `|` separates entries, so neither may appear.
    return s.replace(";", ",").replace("|", "/")

def entries(rows, n):
    return " | ".join(" ; ".join(esc(c) for c in r) for r in rows[:n])

# --- the five shapes --------------------------------------------------------
# 0 spoken description, 1 the id it means HERE, 2 the list that says so,
# 3 the name to say back (nobody says "org.gnome.Nautilus is open").
findable = []
for _ in range(220):
    role = rng.choice([r for r in by_role if len(by_role[r]) >= 2])
    i, n = rng.choice(by_role[role])
    others = [o for o, _ in by_role[role] if o != i]
    findable.append((rng.choice(ROLES[role]), i, installed(must=(i,), forbid=others), n))

# The user names the app; the guessable id is wrong, the list has the real one.
guessy = []
for i, n, _r in apps:
    g = naive(n)
    if g and g != i and len(g) > 2:
        guessy.append((n, g, i, installed(must=(i,), forbid=(g,))))
rng.shuffle(guessy)

# The guess IS the id: one call, no list needed.
easy = [(n, i) for i, n, _ in apps if naive(n) == i]
rng.shuffle(easy)

# Genuinely not installed. The list settles it and the search stops.
absent = []
for n in ["Photoshop", "Microsoft Word", "Excel", "Outlook", "Safari", "iTunes",
          "Final Cut Pro", "Sketch", "Notion", "Figma", "Xcode", "Visual Studio",
          "Premiere", "After Effects", "Illustrator", "OneNote", "Paint.NET"]:
    for _ in range(6):
        g = naive(n)
        absent.append((n, g, installed(forbid=(g, n))))
rng.shuffle(absent)

body = f'''# Launching and controlling desktop applications.
#
# GENERATED by scripts/gen_apps_syn.py from datamix/pools/installed_apps.json —
# this machine's real .desktop ids plus a general pool of the ones a Linux
# desktop commonly carries. Edit the script, not this file.
#
# Tools and their results (nothing else exists):
#   app.launch(name="…")  -> ok | already_running | not_found | failed
#   app.list()            -> the installed ids, comma separated
#   browser.open(url="…") -> ok
#   media.control(action="…") -> ok | nothing_playing
#
# THE INSTALLED SET IS NOT FIXED AND IS NOT KNOWN IN ADVANCE. Every entry below
# carries its own `app.list()` answer, drawn at random and never repeated, and
# real ids are not the words people say: `org.kde.dolphin`, `com.obsproject.Studio`,
# `btrfs-assistant`, `DaVinciResolve`. A model can only answer by READING the
# list it was given. That is the whole point of this file — the version it
# replaces had one seven-id list that appeared in 24 of 25 examples, which
# taught the list as a constant and never produced a dotted id at all.
#
# The policy the templates teach:
#   * the user says a name that IS the id ("blender", "firefox") -> launch it
#   * the user says a name that is not ("Dolphin", "OBS Studio") -> the guess is
#     tried, comes back not_found, and the LIST decides
#   * the user says a description ("the file manager") -> no guess is possible,
#     so `app.list` comes FIRST
#   * not in the list -> say so, and stop. Never guess a second time.

list warm = Done | There you go | Got it | All set

# 0 the spoken description, 1 the id it means in THIS example, 2 the installed
# list, 3 the name to say back. The same description maps to a different id in
# the next entry — which is exactly why the list has to be read.
list findable = {entries(findable, 220)}

# 0 the name spoken, 1 the id a model would guess from it, 2 the real id,
# 3 the installed list.
list guessy = {entries(guessy, 200)}

# The guess happens to be right: no list needed.
list easy = {entries(easy, 60)}

# 0 the name, 1 the guess, 2 a list that does not contain it.
list absent = {entries(absent, 90)}

# A description: nothing to guess from, so the list comes first.
template
user      = {{open|start|launch|pull up}} {{findable.0}}
assistant = <tool>app.list()</tool>
tool      = {{findable.2}}
assistant = <tool>app.launch(name="{{findable.1}}")</tool>
tool      = ok
assistant = {{warm.0}} — {{findable.3}} is {{up|open}}.
category  = apps

template
user      = i need {{findable.0}}{{|, please| if you can}}
assistant = <tool>app.list()</tool>
tool      = {{findable.2}}
assistant = <tool>app.launch(name="{{findable.1}}")</tool>
tool      = ok
assistant = That'd be {{findable.3}} — {{opened it|it's up}}.
category  = apps

# The guess is the id. One call.
template
user      = {{open|start|launch|fire up}} {{easy.0}}
assistant = <tool>app.launch(name="{{easy.1}}")</tool>
tool      = ok
assistant = {{easy.0}} is {{up|open}}.
category  = apps

template
user      = {{open|start}} {{easy.0}}
assistant = <tool>app.launch(name="{{easy.1}}")</tool>
tool      = already_running
assistant = {{Already running|It's already open}} — {{switched you to it|it's there}}.
category  = apps

# The guess is not the id. This is the commonest case on a real desktop, and the
# recovery is mechanical: not_found, list, launch what the list actually says.
template
user              = {{open|start|launch}} {{guessy.0}}
assistant_context = <tool>app.launch(name="{{guessy.1}}")</tool>
tool              = not_found
assistant         = <tool>app.list()</tool>
tool              = {{guessy.3}}
assistant         = <tool>app.launch(name="{{guessy.2}}")</tool>
tool              = ok
assistant         = {{guessy.0}} is {{up|open}} — it goes by `{{guessy.2}}` here.
category          = apps

# Same, but it lists first because it does not want to guess twice.
template
user      = {{can you open|could you start}} {{guessy.0}}?
assistant = <tool>app.list()</tool>
tool      = {{guessy.3}}
assistant = <tool>app.launch(name="{{guessy.2}}")</tool>
tool      = ok
assistant = {{warm.0}} — {{guessy.0}}.
category  = apps

# Not installed. The list ends the search — it does not launch something else.
template
user              = {{open|start|launch}} {{absent.0}}
assistant_context = <tool>app.launch(name="{{absent.1}}")</tool>
tool              = not_found
assistant         = <tool>app.list()</tool>
tool              = {{absent.2}}
assistant         = {{absent.0}} {{isn't installed here|isn't on this machine}}. {{Nothing in the list matches it.|I checked everything available.}}
category          = apps

template
user      = {{do i have|is there}} {{absent.0}}{{ here|}}?
assistant = <tool>app.list()</tool>
tool      = {{absent.2}}
assistant = No — {{absent.0}} {{isn't there|isn't installed}}.
category  = apps

# A dead end is still an offer: it cannot reach the app, but it can reach the web.
template
user              = {{open|start}} {{absent.0}}
assistant_context = <tool>app.launch(name="{{absent.1}}")</tool>
tool              = not_found
assistant         = <tool>app.list()</tool>
tool              = {{absent.2}}
assistant         = Not installed. {{Want me to open it in the browser?|Shall I try the web version?}}
user              = {{yes|please|go on}}
assistant         = <tool>browser.open(url="https://google.com/search?q={{absent.1}}")</tool>
tool              = ok
assistant         = {{Searched for it.|Opened a search.}}
category          = apps

template
user      = what {{apps|programs}} {{do i have|are installed|can you open}}?
assistant = <tool>app.list()</tool>
tool      = {{findable.2}}
assistant = That's what's here.
category  = apps

# The launch simply fails. Nothing to recover to, so it says so.
template
user      = {{open|start}} {{easy.0}}
assistant = <tool>app.launch(name="{{easy.1}}")</tool>
tool      = failed
assistant = {{easy.0}} {{wouldn't start|failed to launch}}. {{Want me to try again?|Something's wrong on its end.}}
category  = apps

list media = play ; play | pause ; pause | stop ; stop | skip ; next | next track ; next

template
user      = {{media.0}}{{| the music| the track| it}}
assistant = <tool>media.control(action="{{media.1}}")</tool>
tool      = ok
assistant = {{Done|Got it}}.
category  = apps

template
user      = {{pause|skip|stop}} {{the music|it}}
assistant = <tool>media.control(action="pause")</tool>
tool      = nothing_playing
assistant = {{Nothing's playing right now|There's nothing playing}}.
category  = apps

list site = github ; https://github.com | youtube ; https://youtube.com | the rust docs ; https://doc.rust-lang.org | my mail ; https://mail.google.com | crates.io ; https://crates.io | hacker news ; https://news.ycombinator.com

template
user      = {{open|pull up}} {{site.0}}{{| in the browser}}
assistant = <tool>browser.open(url="{{site.1}}")</tool>
tool      = ok
assistant = {{There you go|Opened {{site.0}}|Got it}}.
category  = apps
'''
OUT.write_text(body)
print(f"{OUT}: {len(body)} bytes")
print(f"  apps in pool     : {len(apps)}")
print(f"  findable entries : {min(220, len(findable))}")
print(f"  guessy entries   : {min(200, len(guessy))}")
print(f"  easy entries     : {min(60, len(easy))}")
print(f"  absent entries   : {min(90, len(absent))}")
