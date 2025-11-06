# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

# TO-DO: Resize functionality
# if resize:
#     with tempfile.NamedTemporaryFile(
#         dir=output.parent, suffix=output.suffix, delete=False
#     ) as tmp:
#         tmp_path = Path(tmp.name)

#     try:
#         match output.suffix:
#             case ".npy":
#                 with tempfile.NamedTemporaryFile(
#                     dir=output.parent, suffix=".npy", delete=False
#                 ) as tmp:
#                     tmp_path = Path(tmp.name)

#                 resize_mask_memmap(
#                     input_path=str(output),
#                     output_path=str(tmp_path),
#                     target_shape=(image.h, image.w),
#                     row_block=model.tile_width,
#                     col_block=model.tile_height,
#                 )

#                 os.replace(tmp_path, output)

#             case ".zarr":
#                 tmp_dir = tempfile.TemporaryDirectory(dir=output.parent)
#                 tmp_path = Path(tmp_dir.name) / output.name

#                 resize_mask_zarr(
#                     input_path=str(output),
#                     output_path=str(tmp_path),
#                     target_shape=(image.h, image.w),
#                     row_block=model.tile_width,
#                     col_block=model.tile_height,
#                 )

#                 if output.exists():
#                     shutil.rmtree(output)
#                 os.replace(tmp_path, output)

#     except Exception as e:
#         if tmp_path.exists():
#             try:
#                 tmp_path.unlink()
#             except Exception as cleanup_error:
#                 print(f"Failed to clean-up temporary file: {cleanup_error}")
#         raise e
