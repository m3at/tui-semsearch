#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# from rich.console import RenderableType
#
# from rich.syntax import Syntax
# from rich.traceback import Traceback
#
# from textual.app import App
# from textual.widgets import Header, Footer, FileClick, ScrollView, DirectoryTree
#
# from textual_inputs import TextInput
# from textual.message import Message
#
#
# class MyApp(App):
#     """An example of a very simple Textual App"""
#
#     def __init__(self, *args, base_path, **kwargs):
#         self._root = base_path
#         super().__init__(*args, **kwargs)
#
#     async def on_load(self) -> None:
#         """Sent before going in to application mode."""
#
#         # Bind our basic keys
#         await self.bind("b", "view.toggle('sidebar')", "Toggle sidebar")
#         await self.bind("q", "quit", "Quit")
#
#         #
#         await self.bind("enter", "submit", "Submit")
#         await self.bind("escape", "reset_focus", show=False)
#
#         # Get path to show
#         try:
#             self.path = sys.argv[1]
#         except IndexError:
#             self.path = os.path.abspath(os.path.join(os.path.basename(__file__), "../../"))
#
#     async def on_mount(self) -> None:
#         """Call after terminal goes in to application mode"""
#
#         # Create our widgets
#         # In this a scroll view for the code and a directory tree
#         self.body = ScrollView()
#         self.directory = DirectoryTree(self.path, "Code")
#
#         # Dock our widgets
#         await self.view.dock(Header(), edge="top")
#         await self.view.dock(Footer(), edge="bottom")
#
#         #
#         self.username = TextInput(
#             name="username",
#             placeholder="enter your username...",
#             title="Username",
#         )
#         self.username.on_change_handler_name = "handle_username_on_change"
#
#         # Note the directory is also in a scroll view
#         await self.view.dock(self.body, edge="left", size=50)
#         await self.view.dock(ScrollView(self.directory), self.username, edge="top", name="sidebar")
#
#     async def handle_username_on_change(self, message: Message) -> None:
#         self.log(f"Username Field Contains: {message.sender.value}")  # type:ignore
#
#     async def handle_file_click(self, message: FileClick) -> None:
#         """A message sent by the directory tree when a file is clicked."""
#
#         syntax: RenderableType
#         try:
#             # Construct a Syntax object for the path in the message
#             syntax = Syntax.from_path(
#                 message.path,
#                 line_numbers=True,
#                 word_wrap=True,
#                 indent_guides=True,
#                 theme="monokai",
#             )
#         except Exception:
#             # Possibly a binary file
#             # For demonstration purposes we will show the traceback
#             syntax = Traceback(theme="monokai", width=None, show_locals=True)
#         self.app.sub_title = os.path.basename(message.path)
#         await self.body.update(syntax)
#
#     async def action_submit(self) -> None:
#         message = self.username.value
#         self.log(f">>> SUBMIT: {message} {self._root}")
#         await self.body.focus()
#
#     async def action_reset_focus(self) -> None:
#         await self.body.focus()


import os
import json
import pickle
from pathlib import Path
import numpy as np

import coremltools as ct

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x

from tokenizer import SimpleTokenizer


DATA_DIR = Path(os.environ.get("XDG_CONFIG_HOME", default=Path.home() / ".local" / "share")) / "semsearch"


class NeedEmbeddings(BaseException):
    pass


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def l2_sim(emb, query):
    """Euclidean distance (L2 norm)."""
    return 1 - np.linalg.norm((emb[None, :, :] - query[:, None, :]), axis=-1)


def cos_sim(emb, query):
    """Cosine similarity."""
    # Normalize the query, assume the embeddings are already normalized
    query = norm(query)
    return query @ emb.T


def closest_1d(similarity):
    assert similarity.shape[0] == 1
    order = np.argsort(similarity)[..., ::-1][0]
    sim_sorted = similarity[0, order]
    return sim_sorted, order


def get_target_dir(target_dir: Union[str, None]) -> Path:
    path_config = Path(os.environ.get("XDG_CONFIG_HOME", default=Path.home() / ".config")) / "semsearch" / "config.json"
    if path_config.exists():
        with path_config.open("r") as f:
            loaded_dir: Path = Path(json.load(f)["target_dir"])  # type:ignore
            return loaded_dir

    assert target_dir is not None, "Please provide a target directory to scan on first run"

    path_config.parent.mkdir(parents=True, exist_ok=True)
    conf = {"target_dir": target_dir}
    with path_config.open("w") as f:
        json.dump(conf, f)

    return Path(target_dir)


def generate_embeddings(root_text: Path, inferer: Callable) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    files = {}
    list_files = []
    embeddings = []

    for f in tqdm(list(root_text.glob("note_*"))):
        t = f.read_text()

        e = [inferer(x)[0] for x in filter(lambda x: len(x) > 4, t.split("\n")[:-1])]
        if len(e) == 0:
            continue
        # Average embedding per document
        embeddings.append(np.mean(np.array(e), axis=0))
        files[f.name] = os.path.getmtime(f)
        list_files.append(f.name)

    embeddings = norm(embeddings)

    print(f"Elements: {len(files)}")

    _emb = DATA_DIR / "embeddings.npy"
    _map = DATA_DIR / "mapping.pickle"
    np.save(_emb, embeddings)
    with _map.open("wb") as f:
        pickle.dump(
            {
                "files": files,
                "list_files": list_files,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return embeddings, files, list_files


def get_embeddings_if_exist() -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    _emb = DATA_DIR / "embeddings.npy"
    _map = DATA_DIR / "mapping.pickle"
    if not (_emb.exists() and _map.exists()):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raise NeedEmbeddings

    embeddings = np.load(_emb)
    with _map.open("rb") as f:
        t = pickle.load(f)

    return embeddings, t["files"], t["list_files"]


def main(input_path: Union[str, None]):
    print("Setting up...")
    target_dir = get_target_dir(input_path)

    # TODO: download from github release instead
    model_coreml = ct.models.MLModel("./text_encoder_q16.mlmodel")
    # model_coreml = ct.models.MLModel("./text_encoder_q4.mlmodel")

    # Warmup
    print("Warmup...")
    for _ in range(3):
        _ = model_coreml.predict({"input": np.zeros([1, 77], dtype=np.int32)})

    token_black = SimpleTokenizer()
    inferer = lambda x: model_coreml.predict({"input": token_black(x)})["output"]

    try:
        embeddings, files, list_files = get_embeddings_if_exist()
    except NeedEmbeddings:
        embeddings, files, list_files = generate_embeddings(target_dir, inferer)

    # Interractive
    print("go!")
    try:
        while True:
            # text_content = "how to remove a branch in version control"
            text_content = input("> ")
            # text_content = "unix file change"
            # text_content = "cloud services"
            # text_features = norm(inferer(token_black(text_content)))
            text_features = norm(inferer(text_content))

            d = cos_sim(embeddings, text_features)
            dist, nearest = closest_1d(d)

            # print("\n".join(f"[{list_files[x]}] {y:.3f}" for y, x in zip(dist[:5], nearest)))
            for y, x in zip(dist[:3], nearest):
                print(f"~~~ [{list_files[x]}] {y:.3f}")
                with (target_dir / list_files[x]).open() as f:
                    print("".join(str(s) for s, _ in zip(f, range(10))))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="replace_me",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to scan",
    )
    args = parser.parse_args()

    main(args.path)

    # Run our app class
    # MyApp.run(title="Code Viewer", log="/tmp/textual.log", base_path=args.path)
