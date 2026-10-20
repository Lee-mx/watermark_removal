# 水印消除（Windows）

一个在本机处理图片右下角水印的桌面应用。React/TypeScript 负责界面，Python、OpenCV 和 pywebview 负责文件操作及修复。原图不会被修改；输出为原尺寸 PNG。

## 启动

当前图标与渐变修补版请双击 [水印消除.exe](release-icon/水印消除/水印消除.exe)。**请保留 `release-icon/水印消除` 整个文件夹，不要只复制 exe**。程序图标采用 `assets/app_icon_source.png` 提供的图片，已转换为 Windows 图标并用于窗口和安装包。旧版运行期间 Windows 会锁定其中的 DLL，因此关闭旧程序前不能覆盖 `release` 目录。首次运行需要 Windows WebView2 Runtime；若系统缺少，请安装后重试。应用不需要用户安装 Python 或 Node.js。

在本项目开发环境中，也可运行：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'E:\dianping\desktop\run_app.py'
```

## 使用

1. 启动后点击中间加号选择图片，或把图片拖进窗口。支持 JPG、PNG、WebP 多选。
2. 检查每张图片的修复区域。程序会对右下角文字尝试匹配已内置的两个账号模板（`11578066187`、`267777923`）；无法可靠匹配时要求手动圈选。点击“调整区域”可查看或更换模板。匹配并不等于修复效果无痕。
3. 在列表下方选择保存目录，点击“开始去水印”。运行中可取消后续图片。
4. 完成后在下方查看结果，可打开图片或定位文件。失败图片会显示原因。

拖入图片时，浏览器安全限制不暴露原始绝对路径，应用会在本机临时缓存一份，任务结束后可继续预览，关闭应用时清理。单张拖入上限 35 MB；更大的图片请使用加号文件选择。所有图像处理在本机完成，不上传服务器。

## 构建

在 `desktop` 目录执行 `npm install`，然后运行 [build.ps1](build.ps1)。该脚本先构建 React，再通过 PyInstaller 生成 `release/水印消除/水印消除.exe`。构建机需有 Python 3.12，以及 `requirements.txt` 中的依赖；本项目当前开发环境将依赖放在 `.vendor` 和项目根目录的 `.deps` 中。

## 当前范围

- 首版只针对静态图片。水印遮挡的原始像素无法恢复，修复是依据周围画面估算。
- 内置模板只适合匹配的右下角版式，不会保证所有账号文字都完整覆盖；可以通过“调整区域”改用手动选区。水印压在鞋边、锅沿等强边缘上时，即使区域正确也可能留下模糊或断边。
- 账号 `267777923` 使用随程序打包的 [OpenCV LaMa 离线模型](https://huggingface.co/opencv/inpainting_lama)对局部区域补画。手动框选的区域若周边足够平滑，会按四周颜色重建渐变，减少整块色差；复杂背景仍使用边缘感知算法。模型来源及 Apache 2.0 许可见 `desktop/models/LICENSE`。AI 修补一张图片可能需要几十秒；复杂纹理仍需逐张检查，不能保证无痕。
- 输出统一为 PNG，按 `原文件名_clean.png` 命名；重名时自动加数字后缀，不覆盖已有文件。
- 暂未加入自动识别任意位置、任意账号的水印及自动 ZIP 导出。

## 脚本与质量检查

`tools/remove_corner_watermark.py` 与桌面版共用修复算法。可指定 `--method telea` 对照旧方法；`--mask` 使用同账号、同版式的蒙版，`--mask-padding 1` 扩大 1 个原图像素以覆盖文字抗锯齿边缘。不要混用不同账号文字的蒙版。

`desktop/qa/edge-repair/comparison.png` 是盘沿案例的原图、旧算法与新算法局部对照；验证报告记录输出尺寸和蒙版外修改数量。修复只是估算遮挡内容，并非找回原始像素。

白点残留回归案例见 `desktop/qa/footer-coverage/comparison.png`：旧模板复现残留，新模板匹配账号文字并覆盖抗锯齿轮廓。生成对照时会调用桌面实际输出流程，确认与预览一致。对应脚本用法：`tools/remove_corner_watermark.py --mask tools/watermark_mask_11578066187.png --mask-padding 2 --output-dir <输出目录> <原图>`（padding 2 对应本例宽 1280 像素）。
