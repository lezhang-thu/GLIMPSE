**Table:** Cost analysis and token usage statistics on Loghub-2.0 benchmark. Pricing is based on DeepSeek-V3 standards. **GLIMPSE-Serial** achieves the lowest cost due to its superior cache hit rate, while **LUNAR** incurs high communication overhead.

| Method | Input Tokens (Cache Hit)*($0.028/M)* | Input Tokens (Cache Miss)*($0.28/M)* | Output Tokens*($0.42/M)* | Requests | Total Cost ($) |
| :--- | ---: | ---: | :---: | ---: | ---: |
| LUNAR-Serial | 1,550,912 | 400,866 | 195,373 | **3,869** | 0.238 |
| SCULP-Serial | 2,306,368 | 255,621 | 289,801 | 3,173 | 0.258 |
| SCULP-Parallel | 2,302,080 | 255,005 | 287,452 | 3,168 | 0.257 |
| GLIMPSE-Serial | 2,441,600 | **161,520** | 286,845 | 3,276 | **0.234** |
| GLIMPSE-Parallel | 2,336,960 | 274,196 | 296,981 | 3,289 | 0.267 |

