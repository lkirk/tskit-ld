import tskit
from IPython import display

from uuid import uuid4


def print_edge_diffs(
    ts: tskit.TreeSequence,
    tree_indices: set | None = None,
    **draw_kwargs,
) -> None:
    trees = range(ts.num_trees) if tree_indices is None else tree_indices
    grid_style = [
        "display:grid",
        # fixed num of columns, we can add more rows
        "width:max-content",
        "overflow-x:auto",
        "background: #eee",
        "padding:5px",
    ]

    add = "{stroke: cyan; stroke-width: 2px}"
    rem = "{stroke: red; stroke-width: 2px}"
    mask = "{stroke-width: 0px}"
    for tree_idx, (_, edges_out, edges_in) in enumerate(ts.edge_diffs()):
        if tree_idx not in trees:
            continue
        if tree_idx == 0:
            display.display(display.Markdown(f"### Tree {tree_idx}"))
        else:
            display.display(
                display.Markdown(f"### Tree {tree_idx - 1} ->  Tree {tree_idx}")
            )

        masked = []
        divs = []
        tree_styles = []
        tree_grid_style = ";".join(
            grid_style
            + [
                f"grid-template-columns:repeat({len(edges_out) + len(edges_in)}, minmax(0, auto))"
            ]
        )
        for e in edges_out:
            tree = ts.at_index(tree_idx - 1)
            div_id = uuid4()
            tree_styles.append(
                f"#id-{div_id} svg .a{e.parent}.n{e.child} > .edge {rem}"
            )
            divs.append(f'<div id="id-{div_id}">{tree.draw_svg(**draw_kwargs)}</div>')
            for me in masked:
                tree_styles.append(
                    f"#id-{div_id} svg .a{me.parent}.n{me.child} > .edge {mask}"
                )
            masked.append(e)

        masked = list(reversed(edges_in))
        for e in edges_in:
            tree = ts.at_index(tree_idx)
            div_id = uuid4()
            tree_styles.append(
                f"#id-{div_id} svg .a{e.parent}.n{e.child} > .edge {add}"
            )
            masked.pop()
            for me in masked:
                tree_styles.append(
                    f"#id-{div_id} svg .a{me.parent}.n{me.child} > .edge {mask}"
                )
            divs.append(f'<div id="id-{div_id}">{tree.draw_svg(**draw_kwargs)}</div>')
        display.display(
            display.HTML(
                f'<div style="{tree_grid_style}"><style>{" ".join(tree_styles)}</style>{"".join(divs)}</div>'
            )
        )


def print_cumulative_edge_diffs(ts: tskit.TreeSequence, **draw_kwargs) -> None:
    style = ""
    for tree_idx, (_, edges_out, edges_in) in enumerate(ts.edge_diffs()):
        if tree_idx == 0:
            continue
        for e in edges_out:
            style += f".t{tree_idx - 1} .a{e.parent}.n{e.child} > .edge {{stroke: red; stroke-width: 2px}}"
        for e in edges_in:
            style += f".t{tree_idx} .a{e.parent}.n{e.child} > .edge {{stroke: cyan; stroke-width: 2px}}"
    display.display(ts.draw_svg(style=style, **draw_kwargs))
