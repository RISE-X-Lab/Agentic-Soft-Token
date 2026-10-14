# Third-party notices

`src/stable_interfaces/runtime/dynamics.py` adapts kernel orchestration from flash-linear-attention
0.4.1 (https://github.com/fla-org/flash-linear-attention), not its kernels.
The unweighted-innovation path additionally invokes the existing chunk read
kernel with shifted-key queries and an FP32 output buffer. Installed kernel source
is unchanged; the extra read is orchestrated locally.

MIT License

Copyright (c) 2023-2025 Songlin Yang, Yu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The Qwen bridge uses the installed Transformers 5.2.0 modules and their public
weights. GDN preprocessing follows its native forward order (Apache-2.0;
https://github.com/huggingface/transformers). It does not modify site-packages.

Original code, documentation, and bundled calibration are covered by the MIT
License in `LICENSE`. Model weights and benchmark datasets are not distributed
here; obtain them separately under their respective licenses.
