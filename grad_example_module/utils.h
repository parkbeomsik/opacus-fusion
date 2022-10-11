#pragma once

#define LOG_STDERR(s, verbose) { if (verbose) { std::cerr << s << "\n";} }

#define TIME_PROFILE(v, profile) {\
    if (time_profile) {  \
    torch::cuda::synchronize();\
    v += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;\
    start_time = std::chrono::high_resolution_clock::now();\
    }\
}
