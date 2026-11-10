# mkcorpus_packs.py OUTDIR: one text per task (python, rust, c, web, php, go, java, ts) for the expert packs:
# source files of well-known open projects (local stdlib / cargo sources, jsDelivr for the rest) interleaved with
# short task prompts in that language; split by file into <task>_fit.txt (~200 KB) and <task>_held.txt (~60 KB).
import glob, os, random, sys, urllib.request

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
J = "https://cdn.jsdelivr.net/gh/"
N = "https://cdn.jsdelivr.net/npm/"

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            b = r.read()
        return b.decode("utf-8", "replace")
    except Exception as e:
        print("  miss", url, e, flush=True)
        return None

def local(patterns, cap=400):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p, recursive=True))
    files = [f for f in files if "/test" not in f and os.path.getsize(f) > 2000]
    random.Random(7).shuffle(files)
    res = []
    for f in files[:cap]:
        try:
            res.append((f, open(f, encoding="utf-8", errors="replace").read()))
        except Exception:
            pass
    return res

def remote(base, paths):
    res = []
    for p in paths:
        t = fetch(base + p)
        if t:
            res.append((p, t))
    return res

three = J + "mrdoob/three.js@r160/"
boot = J + "twbs/bootstrap@v5.3.3/site/content/docs/5.3/examples/"
wp = J + "WordPress/WordPress@6.4.2/"
lv = J + "laravel/framework@v10.48.4/src/Illuminate/"
go = J + "golang/go@go1.21.5/src/"
cl = J + "apache/commons-lang@rel/commons-lang-3.14.0/src/main/java/org/apache/commons/lang3/"
gs = J + "google/gson@gson-parent-2.10.1/gson/src/main/java/com/google/gson/"
vs = J + "microsoft/vscode@1.85.0/src/vs/base/common/"
rd = J + "redis/redis@7.2.3/src/"
sq = J + "sqlite/sqlite@version-3.44.2/src/"

sources = {
    "python": lambda: local(["/usr/lib/python3*/**/*.py"]),
    "rust": lambda: local([os.path.expanduser("~/.cargo/registry/src/*/*/src/**/*.rs")]),
    "c": lambda: remote(rd, ["server.c", "networking.c", "dict.c", "t_string.c", "sds.c", "object.c", "db.c", "util.c", "t_list.c", "rdb.c"])
         + remote(sq, ["btree.c", "pager.c", "select.c", "where.c", "expr.c", "func.c", "util.c", "printf.c"])
         + local(["/usr/include/*.h"], cap=60),
    "web": lambda: remote(three + "examples/", [f"{n}.html" for n in [
            "webgl_animation_keyframes", "webgl_geometry_cube", "webgl_materials_standard", "webgl_loader_gltf",
            "webgl_postprocessing_unreal_bloom", "webgl_shadowmap", "webgl_water", "webgl_lights_hemisphere", "webgl_camera",
            "webgl_buffergeometry", "webgl_instancing_performance", "webgl_points_sprites", "css3d_periodictable",
            "misc_controls_orbit", "webgl_materials_envmaps", "webgl_animation_skinning_blending", "webgl_geometry_text",
            "webgl_interactive_cubes", "webgl_morphtargets", "webgl_ocean"]])
         + remote(boot, [f"{n}/index.html" for n in ["album", "blog", "checkout", "dashboard", "pricing", "product",
            "sign-in", "carousel", "cover", "features", "headers", "heroes", "sidebars", "jumbotron"]])
         + remote(three + "src/", ["core/Object3D.js", "math/Vector3.js", "math/Matrix4.js", "loaders/FileLoader.js",
            "cameras/PerspectiveCamera.js", "materials/Material.js", "core/BufferGeometry.js"])
         + remote(N, ["bootstrap@5.3.3/dist/css/bootstrap.css", "normalize.css@8.0.1/normalize.css"]),
    "php": lambda: remote(wp, ["wp-includes/formatting.php", "wp-includes/post.php", "wp-includes/class-wp-query.php",
            "wp-includes/functions.php", "wp-includes/plugin.php", "wp-includes/user.php", "wp-includes/general-template.php",
            "wp-includes/link-template.php", "wp-includes/kses.php", "wp-includes/class-wp-user.php", "wp-admin/includes/file.php"])
         + remote(lv, ["Support/Str.php", "Collections/Collection.php", "Routing/Router.php", "Http/Request.php",
            "Database/Query/Builder.php", "Validation/Validator.php", "Support/Arr.php"]),
    "go": lambda: remote(go, ["net/http/server.go", "fmt/print.go", "strings/strings.go", "sort/sort.go",
            "encoding/json/encode.go", "encoding/json/decode.go", "os/file.go", "bufio/bufio.go", "time/format.go",
            "net/url/url.go", "bytes/buffer.go", "path/filepath/path.go", "sync/mutex.go", "context/context.go",
            "net/http/client.go", "io/io.go"]),
    "java": lambda: remote(cl, ["StringUtils.java", "ArrayUtils.java", "ObjectUtils.java", "BooleanUtils.java", "CharUtils.java",
            "Validate.java", "time/DateUtils.java", "math/NumberUtils.java", "builder/ToStringBuilder.java", "reflect/MethodUtils.java"])
         + remote(gs, ["Gson.java", "JsonParser.java", "stream/JsonReader.java", "stream/JsonWriter.java", "JsonObject.java", "GsonBuilder.java"]),
    "ts": lambda: remote(vs, ["strings.ts", "arrays.ts", "async.ts", "event.ts", "lifecycle.ts", "uri.ts", "objects.ts",
            "types.ts", "path.ts", "json.ts", "network.ts", "cancellation.ts", "errors.ts", "map.ts", "iterator.ts"]),
}

lang_name = {"python": "Python", "rust": "Rust", "c": "C", "web": "HTML/CSS/JavaScript", "php": "PHP", "go": "Go",
             "java": "Java", "ts": "TypeScript"}
tasks = ["parses a CSV file and returns the rows as records", "debounces a callback", "implements an LRU cache",
         "validates an email address", "reads a JSON config with defaults", "computes the median of a list",
         "retries an HTTP request with exponential backoff", "merges two sorted arrays", "slugifies a title",
         "paginates a list of items", "builds a binary search tree with insert and search", "rate-limits calls per second",
         "formats a byte count as KB/MB/GB", "walks a directory tree and lists files by size", "escapes HTML entities",
         "groups records by a key", "parses command-line flags", "computes a SHA-256 checksum of a file",
         "renders a table as Markdown", "deduplicates a list while keeping order"]
forms = ["Write a {L} function that {t}.", "In {L}, implement code that {t}. Add brief comments.",
         "Fix and improve this {L} code so it {t}:", "Explain how to write {L} code that {t}, then show it."]
web_tasks = ["a responsive landing page with a hero section and three feature cards", "a login form with validation",
             "a three.js scene with a rotating cube and orbit controls", "a CSS grid photo gallery with hover zoom",
             "a dark-mode toggle that remembers the choice", "an animated SVG loading spinner", "a todo list in vanilla JS",
             "a canvas particle animation", "a modal dialog with a fade-in", "a pricing table with three plans"]

def prompts(task):
    r = random.Random(sum(map(ord, task)))
    L = lang_name[task]
    out = []
    if task == "web":
        for w in web_tasks:
            out.append(r.choice(["Create {w} as a single HTML file.", "Build {w} using HTML, CSS and JavaScript.",
                                 "Write the code for {w}."]).format(w=w))
    for t in tasks:
        out.append(r.choice(forms).format(L=L, t=t))
    r.shuffle(out)
    return out

for task, src in sources.items():
    files = src()
    total = sum(len(t) for _, t in files)
    print(f"{task}: {len(files)} files, {total} chars", flush=True)
    if not files:
        continue
    ps = prompts(task)
    fit, held = [], []
    fit_n = held_n = 0
    for i, (name, text) in enumerate(files):
        text = text[:24000]
        p = ps[i % len(ps)]
        piece = f"{p}\n\n{text}\n\n"
        if i % 4 == 3 and held_n < 60000:
            held.append(piece); held_n += len(piece)
        elif fit_n < 200000:
            fit.append(piece); fit_n += len(piece)
    open(f"{out}/{task}_fit.txt", "w").write("".join(fit))
    open(f"{out}/{task}_held.txt", "w").write("".join(held))
    print(f"  fit {fit_n} chars, held {held_n} chars", flush=True)
print("CORPUS_DONE")
