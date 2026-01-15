# TO-DO: Refactor visualization methods.
# def show(
#         self,
#         ax: None | plt.Axes = None,
#         figsize: None | tuple[int, int] = None,
#         edgecolor: str | None = "black",
#         facecolor: str | None = "green",
#         linewidth: int | float = 1,
#         alpha: int | float = 0.1,
#         t: int = 5000,
#         force: bool = False,
#         height: int = None,
#         width: int = None,
#         resample: str | None = "bilinear",
#         hide_axes: bool = True,
#         save_fig: str | Path = None,
#     ) -> plt.Axes:
#         """Visualize tile overlaid on image thumbnail.
#
#         Parameters
#         ----------
#         ax : None | plt.Axes
#             Optional matplot axes object.
#         figsize : None | tuple[int, int]
#             Option figure size.
#         edgecolor : str | None
#             Edge color around plotted tiles.
#         facecolor : str | None
#             Face color of plotted tiles.
#         linewidth : int | float
#             Edge line width around plotted tiles.
#         t: int
#             Find thumbnail with height + width closest to this value.
#         force: bool
#             If True and image is not pyramidal, then the thumbnail will be
#             generated at the height and width of the full resolution image.
#         height: None | int
#             Optionally resize the image thumbnail to a user-provided height.
#         width: None | int
#             Optionally resize the image thumbnail to a user-provided width.
#         resample: str | None
#             Resampling filter ('bilinear', 'bicubic', 'lanczos', 'nearest').
#         alpha : int | float
#             Alpha of tile colors.
#         hide_axes : bool
#             If True, then all x-axis, y-axis, and spines are removed.
#         save_fig : str | Path
#             Save figure to specified path.
#
#         Returns
#         -------
#         plt.Axes
#             Matplotlib axes object of tile.
#         """
#         thumbnail = self.image.thumbnail(
#             t=t, force=force, height=height, width=width, resample=resample
#         ).to_numpy()
#
#         ih, iw, _ = self.image.shape
#         th, tw, _ = thumbnail.shape
#         scale = th / ih
#
#         if ax is None:
#             fig, ax = plt.subplots(figsize=figsize)
#
#         coordinates, self.coordinates = itertools.tee(self.coordinates, 2)
#
#         ax.imshow(thumbnail)
#         for coord in coordinates:
#             x, y, w, h, *_ = coord
#             x = int(x * scale)
#             y = int(y * scale)
#             w = int(w * scale)
#             h = int(h * scale)
#             rect = plt.Rectangle(
#                 (x, y),
#                 w,
#                 h,
#                 edgecolor=edgecolor,
#                 facecolor=facecolor,
#                 linewidth=linewidth,
#                 alpha=alpha,
#             )
#             ax.add_patch(rect)
#
#         if hide_axes:
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#             for pos in ["left", "right", "top", "bottom"]:
#                 ax.spines[pos].set_visible(False)
#
#         if isinstance(save_fig, (str, Path)):
#             plt.tight_layout()
#             plt.savefig(save_fig, bbox_inches="tight")
#         else:
#             return ax
