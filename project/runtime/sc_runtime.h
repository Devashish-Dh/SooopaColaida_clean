#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Print all currently available SuperCollider reports immediately. The runtime
// also invokes this automatically at normal process exit after observing the
// first CUDA module load.
void sc_report_now(void);

#ifdef __cplusplus
}
#endif
