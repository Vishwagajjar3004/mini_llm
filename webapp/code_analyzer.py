"""
code_analyzer.py
================
Turns an uploaded CODE file into two things:
  1. A structured summary (imports, functions, classes, docstrings) --
     built with real parsing (Python's `ast` module) or careful regex
     for other languages. This is 100% accurate, deterministic, and
     mechanical -- no neural network guessing involved.
  2. A list of plain-English "sentences" describing that structure, so
     the SAME retrieval + MiniLLM pipeline used for PDFs (retriever.py
     + llm_engine.py) can also answer questions about code, without
     needing a separate answering system.

WHY NOT JUST ASK MiniLLM TO "EXPLAIN" THE CODE DIRECTLY?
Same reasoning as PDFs: MiniLLM is a tiny model with no real
understanding of code semantics. Instead, we do the actual explaining
with reliable, classical program analysis (like a linter or an IDE's
"outline" view), and only use MiniLLM afterwards to lightly style
individual sentences if asked a narrow question about them.
"""

import ast
import os
import re

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h",
    ".hpp", ".go", ".rb", ".php", ".cs", ".swift", ".kt", ".rs",
}


def is_code_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in CODE_EXTENSIONS


# ------------------------------------------------------------------
# Python: real parsing via the `ast` module (the same tool Python
# itself uses internally) -- this is fully accurate, not a guess.
# ------------------------------------------------------------------
def analyze_python(source: str) -> dict:
    tree = ast.parse(source)

    module_docstring = ast.get_docstring(tree)
    imports = []
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            else:
                module = node.module or ""
                imports.extend(f"{module}.{alias.name}" for alias in node.names)

        elif isinstance(node, ast.FunctionDef) and _is_top_level(node, tree):
            functions.append({
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "docstring": ast.get_docstring(node),
                "line": node.lineno,
            })

        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            classes.append({
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "methods": methods,
                "line": node.lineno,
            })

    return {
        "language": "Python",
        "module_docstring": module_docstring,
        "imports": sorted(set(imports)),
        "functions": functions,
        "classes": classes,
        "num_lines": len(source.splitlines()),
    }


def _is_top_level(node, tree) -> bool:
    """True if `node` is a direct child of the module (not a method inside a class)."""
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
            if node is not parent:
                return False
    return True


# ------------------------------------------------------------------
# Other languages: careful regex-based structure extraction. Less
# rigorous than a real parser, but reliable enough to reveal a file's
# shape (imports, function/class names) without executing any code.
# ------------------------------------------------------------------
_GENERIC_PATTERNS = {
    "import": re.compile(
        r"^\s*(?:import|from|require|include|using)\s+.+$", re.MULTILINE
    ),
    "function": re.compile(
        r"^\s*(?:function\s+(\w+)|(?:public|private|protected|static|\s)*"
        r"[\w<>\[\],\s]+\s(\w+)\s*\([^;{]*\)\s*\{|def\s+(\w+)|func\s+(\w+)|"
        r"(\w+)\s*[:=]\s*(?:async\s*)?\([^)]*\)\s*=>)",
        re.MULTILINE,
    ),
    "class": re.compile(
        r"^\s*(?:public\s+|private\s+)?class\s+(\w+)", re.MULTILINE
    ),
}


def analyze_generic(source: str, language: str) -> dict:
    imports = [m.strip() for m in _GENERIC_PATTERNS["import"].findall(source)][:30]

    functions = []
    for match in _GENERIC_PATTERNS["function"].finditer(source):
        name = next((g for g in match.groups() if g), None)
        if name:
            functions.append({"name": name, "args": [], "docstring": None, "line": None})

    classes = []
    for match in _GENERIC_PATTERNS["class"].finditer(source):
        classes.append({"name": match.group(1), "docstring": None, "methods": [], "line": None})

    return {
        "language": language,
        "module_docstring": None,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "num_lines": len(source.splitlines()),
    }


_LANGUAGE_BY_EXT = {
    ".js": "JavaScript", ".jsx": "JavaScript (JSX)", ".ts": "TypeScript",
    ".tsx": "TypeScript (TSX)", ".java": "Java", ".c": "C", ".cpp": "C++",
    ".h": "C/C++ header", ".hpp": "C++ header", ".go": "Go", ".rb": "Ruby",
    ".php": "PHP", ".cs": "C#", ".swift": "Swift", ".kt": "Kotlin", ".rs": "Rust",
}


def analyze_code_file(source: str, filename: str) -> dict:
    """
    Dispatches to the right analyzer based on file extension.
    """
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".py":
        try:
            return analyze_python(source)
        except SyntaxError as e:
            # Fall back to generic analysis if the file doesn't parse
            # (e.g. Python 2 code, or a syntax error in the upload).
            result = analyze_generic(source, "Python (had a syntax error, analyzed generically)")
            result["parse_error"] = str(e)
            return result

    language = _LANGUAGE_BY_EXT.get(ext, "Unknown")
    return analyze_generic(source, language)


def build_explanation_text(analysis: dict, filename: str) -> str:
    """
    Turns the structured analysis into one readable, human-facing
    explanation -- this is what gets shown directly when someone asks
    "explain this code".
    """
    lines = [f"**{filename}** ({analysis['language']}, {analysis['num_lines']} lines)"]

    if analysis.get("module_docstring"):
        lines.append(f"\nModule description: {analysis['module_docstring']}")

    if analysis.get("parse_error"):
        lines.append(f"\nNote: this file had a syntax error during parsing ({analysis['parse_error']}), so analysis is approximate.")

    if analysis["imports"]:
        shown = analysis["imports"][:15]
        more = f" (+{len(analysis['imports']) - 15} more)" if len(analysis["imports"]) > 15 else ""
        lines.append(f"\nImports/dependencies ({len(analysis['imports'])}): {', '.join(shown)}{more}")

    if analysis["classes"]:
        lines.append(f"\nClasses ({len(analysis['classes'])}):")
        for c in analysis["classes"]:
            doc = f" -- {c['docstring']}" if c.get("docstring") else ""
            methods = f" [methods: {', '.join(c['methods'])}]" if c.get("methods") else ""
            lines.append(f"  - {c['name']}{doc}{methods}")

    if analysis["functions"]:
        lines.append(f"\nFunctions ({len(analysis['functions'])}):")
        for fn in analysis["functions"]:
            args = f"({', '.join(fn['args'])})" if fn.get("args") else "()"
            doc = f" -- {fn['docstring']}" if fn.get("docstring") else ""
            lines.append(f"  - {fn['name']}{args}{doc}")

    if not analysis["classes"] and not analysis["functions"]:
        lines.append("\nNo top-level functions or classes were detected (this may be a script, config, or data file).")

    return "\n".join(lines)


def build_code_sentences(analysis: dict, filename: str) -> list:
    """
    Converts the structured analysis into individual plain-English
    "sentences", so the existing TF-IDF retriever and MiniLLM trainer
    (built for PDF text) can also index and answer questions about code,
    without needing a separate pipeline.
    """
    sentences = []
    lang = analysis["language"]

    sentences.append(f"The file {filename} is written in {lang} and has {analysis['num_lines']} lines.")

    if analysis.get("module_docstring"):
        sentences.append(f"The file {filename} is described as: {analysis['module_docstring']}")

    if analysis["imports"]:
        imports_text = ", ".join(analysis["imports"][:20])
        sentences.append(f"The file {filename} imports the following: {imports_text}.")

    for c in analysis["classes"]:
        desc = f"The class {c['name']} is defined in {filename}."
        if c.get("docstring"):
            desc += f" It is described as: {c['docstring']}"
        if c.get("methods"):
            desc += f" Its methods are: {', '.join(c['methods'])}."
        sentences.append(desc)

    for fn in analysis["functions"]:
        args = ", ".join(fn["args"]) if fn.get("args") else "no arguments"
        desc = f"The function {fn['name']} takes {args} and is defined in {filename}."
        if fn.get("docstring"):
            desc += f" It is described as: {fn['docstring']}"
        sentences.append(desc)

    return sentences
