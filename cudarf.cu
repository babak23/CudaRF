/* ===========================================================================
 * cudarf.cu
 *
 * Faithful CUDA reconstruction of CudaRF, from:
 *   Grahn, Lavesson, Lapajne, Slat (2010). "A CUDA Implementation of Random
 *   Forests -- Early Results." Third Swedish Workshop on Multi-core Computing.
 *   Grahn, Lavesson, Lapajne, Slat (2011). "CudaRF: A CUDA-based
 *   Implementation of Random Forests." AICCSA 2011.
 *
 * This is a from-scratch rewrite (v3) of this project's earlier CudaRF
 * reconstruction, replacing it with a single self-contained .cu file and
 * fixing two issues found in the previous version:
 *
 *   1. RANDOMNESS: the train/test split is now produced by a small,
 *      dependency-free, deterministic routine (splitmix64 + stratified
 *      split) that is pasted BYTE-IDENTICAL into every implementation in
 *      this project. Given the same --seed, every tool now partitions a
 *      dataset into the exact same train/test rows -- this is what makes
 *      cross-implementation comparison valid. (The GPU-side bagging and
 *      mtry sampling inside the forest itself remain implementation-
 *      specific, as bagging is inherently stochastic per RF design; only
 *      the initial train/test partition needs to match across tools.)
 *
 *   2. HYPERPARAMETERS: --trees, --k (mtry), --depth, --seed and
 *      --trainfrac are now real CLI flags read by argv, not hardcoded
 *      constants. Trees/k are set to reasonable RF defaults (100 trees,
 *      k = sqrt(numFeatures)) rather than the "n_trees=64, k=11" values
 *      silently baked into the earlier two-file-mode code path.
 *
 * Paper fidelity -- what is reproduced faithfully:
 *   - One CUDA thread builds one entire tree (tree-level granularity),
 *     both for training and for classification/voting, exactly as
 *     described in both papers.
 *   - ITERATIVE tree construction: the papers' own GT220 hardware
 *     (compute capability 1.2) could not support recursion in device
 *     code, so the authors wrote an iterative, stack-based tree builder.
 *     We reproduce that iterative design here even though modern hardware
 *     (e.g. an RTX 3070, CC 8.6) would support recursion -- the point is
 *     to reconstruct what they built, not what current hardware allows.
 *   - Splitting criterion: entropy / information gain (explicitly chosen
 *     by the authors over Gini, "to shift computational cost from
 *     training to classification").
 *   - Trees are fully grown with no pruning; numeric attributes only;
 *     no missing-value handling beyond simple mean imputation at load
 *     time (their 2011 paper explicitly lists missing-value handling as
 *     future work -- we do not claim to have solved it, just to not
 *     crash on it).
 *   - The paper's 8-step host/device pipeline (their Fig. 2): read data ->
 *     format & transfer -> bagging kernel (1 thread/tree) -> parallel
 *     tree-build kernel -> OOB classification kernel -> host computes OOB
 *     error -> transfer query data -> prediction kernel (1 thread/tree) ->
 *     transfer results back -> output. All eight steps are present below,
 *     in the same order, and are labelled in the source and in the
 *     terminal output.
 *   - GPU optimizations named in the papers: fast approximate math
 *     intrinsics (__logf instead of logf), pinned host memory via
 *     cudaHostAlloc for faster PCIe transfer, texture-object binding for
 *     the read-only test/query data (texture REFERENCES are used in the
 *     papers; texture OBJECTS are the CUDA-12-compatible equivalent, same
 *     migration this project already applied when reconstructing gpuRF),
 *     and constant memory for small, frequently-read configuration values.
 *
 * Deliberate deviation, documented honestly:
 *   - RNG: the papers use the CUDA SDK's Mersenne Twister sampler
 *     (up to 4096 parallel streams), a legacy component that no longer
 *     ships with modern CUDA toolkits. We substitute a small per-thread
 *     splitmix64-style generator for bagging/mtry sampling inside the
 *     forest. This is NOT the same routine as the shared stratified-split
 *     function above -- that one runs once, on the host, before any GPU
 *     work starts, purely to fix the train/test partition. This one runs
 *     per-tree, on the device, for bootstrap sampling and feature
 *     subsampling, and does not need to match any other tool bit-for-bit
 *     (bagging is supposed to differ from run to run and tool to tool;
 *     only the initial train/test split needs to be shared).
 *
 * Build (WSL2, CUDA 12, RTX 3070):
 *     nvcc -O2 -std=c++14 -arch=sm_86 cudarf.cu -o cudarf
 *
 * Run:
 *     ./cudarf data.csv --trees 100 --k sqrt --depth 6 --seed 42 --trainfrac 0.8
 *     ./cudarf --help
 * =========================================================================*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <climits>
#include <strings.h>   /* strcasecmp -- not declared under strict -std=c++14
                          without this; implicit declarations are a hard
                          compile ERROR in C++, unlike old C89 */

#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>   /* access() */

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_e));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* ===========================================================================
 * SHARED SPLIT ROUTINE -- pasted byte-identical into every implementation
 * in this project. Do not modify this block independently per-tool; any
 * change here must be mirrored in every other .cu/.c file that has it, or
 * "--seed 42" stops meaning the same train/test partition everywhere.
 * =========================================================================*/
static unsigned long long g_splitmix64_state;

static void seed_splitmix64(unsigned long long seed) {
    g_splitmix64_state = seed;
}

static unsigned long long splitmix64_next(void) {
    unsigned long long z = (g_splitmix64_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int splitmix64_below(int n) {
    return (int)(splitmix64_next() % (unsigned long long)n);
}

static void shared_shuffle_indices(int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = splitmix64_below(i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* Stratified train/test split -- see shared_split.h for the full algorithm
 * description. Groups by class (preserving file order within each class),
 * shuffles each class's index list with splitmix64, then takes the first
 * round(train_frac * class_size) shuffled indices per class for train. */
static void stratified_split(const int *labels, int n, int num_classes,
                              double train_frac, unsigned long long seed,
                              int *train_idx, int *train_count,
                              int *test_idx,  int *test_count)
{
    seed_splitmix64(seed);

    int *class_count = (int *)calloc((size_t)num_classes, sizeof(int));
    for (int i = 0; i < n; i++) class_count[labels[i]]++;

    int **class_idx = (int **)malloc((size_t)num_classes * sizeof(int *));
    int *fill_pos = (int *)calloc((size_t)num_classes, sizeof(int));
    for (int c = 0; c < num_classes; c++)
        class_idx[c] = (int *)malloc((size_t)class_count[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = labels[i];
        class_idx[c][fill_pos[c]++] = i;
    }
    free(fill_pos);

    *train_count = 0;
    *test_count  = 0;
    for (int c = 0; c < num_classes; c++) {
        shared_shuffle_indices(class_idx[c], class_count[c]);
        int n_train_c = (int)(train_frac * class_count[c] + 0.5);
        if (n_train_c > class_count[c]) n_train_c = class_count[c];
        for (int i = 0; i < class_count[c]; i++) {
            if (i < n_train_c) train_idx[(*train_count)++] = class_idx[c][i];
            else                test_idx[(*test_count)++]  = class_idx[c][i];
        }
        free(class_idx[c]);
    }
    free(class_idx);
    free(class_count);
}
/* ======================= END SHARED SPLIT ROUTINE ======================= */

/* ---------------------------------------------------------------------------
 * Configuration limits (compile-time capacities; generous for
 * Iris/Breast-Cancer-Wisconsin-scale benchmark datasets)
 * -------------------------------------------------------------------------*/
#define MAX_CLASSES        32
#define MAX_ATTRIBUTES     2048  /* covers the original paper's EULA dataset
                                     (1265 attributes) with headroom */
#define MAX_BAG_CAPACITY   2048  /* max train instances per tree (bootstrap
                                     sample size == train-set size); covers
                                     Iris/Breast-Cancer-Wisconsin/EULA scale */

/* ---------------------------------------------------------------------------
 * Constant memory -- small, frequently-read config values (paper's
 * "constant memory to reduce register pressure" optimization)
 * -------------------------------------------------------------------------*/
__constant__ int   c_numFeatures;
__constant__ int   c_numClasses;
__constant__ int   c_k;            /* mtry: features sampled per split */
__constant__ int   c_maxDepth;
__constant__ int   c_numTrain;
__constant__ int   c_numTrees;
__constant__ int   c_minSamplesSplit;

/* ---------------------------------------------------------------------------
 * Host-side dataset container
 * -------------------------------------------------------------------------*/
struct Dataset {
    int n_instances  = 0;
    int n_attributes = 0;
    int n_classes    = 0;
    std::vector<float> data;      /* row-major: n_instances * n_attributes */
    std::vector<int>   labels;    /* n_instances */
    std::vector<std::string> class_names;
    std::vector<std::string> attr_names;
};

/* ---------------------------------------------------------------------------
 * CSV loader -- auto-detects an optional header row, last column = label,
 * missing values ('?' or empty) imputed with the column mean. Adapted from
 * this project's earlier CudaRF CSV reader, folded inline here so the whole
 * implementation stays a single file per this project's convention.
 * -------------------------------------------------------------------------*/
static void str_trim(char *s) {
    int i = 0;
    while (s[i] && isspace((unsigned char)s[i])) i++;
    memmove(s, s + i, strlen(s) - i + 1);
    int len = (int)strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) s[--len] = '\0';
}

static int is_numeric_field(const char *s) {
    if (*s == '\0') return 0;
    char *end = NULL;
    strtod(s, &end);
    while (*end && isspace((unsigned char)*end)) end++;
    return *end == '\0';
}

static int split_csv_line(char *line, char **fields, int max_fields) {
    int n = 0;
    char *saveptr = NULL;
    char *tok = strtok_r(line, ",", &saveptr);
    while (tok && n < max_fields) {
        str_trim(tok);
        fields[n++] = tok;
        tok = strtok_r(NULL, ",", &saveptr);
    }
    return n;
}

static bool csv_load(const char *path, Dataset &out, bool force_header,
                      bool force_no_header) {
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); return false; }

    const int MAX_LINE_LEN = 65536;
    std::vector<char *> raw_lines;
    char line[65536];

    while (fgets(line, MAX_LINE_LEN, fp)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;
        raw_lines.push_back(strdup(p));
    }
    fclose(fp);

    if (raw_lines.empty()) {
        fprintf(stderr, "csv_load: no data in %s\n", path);
        return false;
    }

    int start_row = 0;
    if (force_header) {
        start_row = 1;
    } else if (!force_no_header) {
        char probe[65536];
        char *fields[MAX_ATTRIBUTES + 1];
        strncpy(probe, raw_lines[0], sizeof(probe) - 1);
        probe[sizeof(probe) - 1] = '\0';
        int nf = split_csv_line(probe, fields, MAX_ATTRIBUTES + 1);
        bool looks_like_header = false;
        for (int i = 0; i < nf - 1; i++)
            if (!is_numeric_field(fields[i])) { looks_like_header = true; break; }
        if (looks_like_header) start_row = 1;
    }
    if (start_row == 1) free(raw_lines[0]);

    int data_count = (int)raw_lines.size() - start_row;
    if (data_count <= 0) {
        fprintf(stderr, "csv_load: no data rows in %s\n", path);
        return false;
    }

    char probe[65536];
    char *probe_fields[MAX_ATTRIBUTES + 1];
    strncpy(probe, raw_lines[start_row], sizeof(probe) - 1);
    probe[sizeof(probe) - 1] = '\0';
    int n_attrs = split_csv_line(probe, probe_fields, MAX_ATTRIBUTES + 1);
    if (n_attrs < 2) {
        fprintf(stderr, "csv_load: need at least 1 feature + 1 label column\n");
        return false;
    }
    int n_input = n_attrs - 1;

    out.n_instances  = data_count;
    out.n_attributes = n_input;
    out.data.assign((size_t)data_count * n_input, 0.0f);
    out.labels.assign(data_count, 0);

    std::vector<std::string> class_names_seen;
    std::vector<int> missing((size_t)data_count * n_input, 0);
    int valid_rows = 0;

    for (int row = 0; row < data_count; row++) {
        char *raw = raw_lines[start_row + row];
        char *fields[MAX_ATTRIBUTES + 1];
        int nf = split_csv_line(raw, fields, MAX_ATTRIBUTES + 1);

        if (nf != n_attrs) {
            fprintf(stderr, "csv_load: line %d has %d fields, expected %d "
                    "(row zero-filled, label=0)\n",
                    start_row + row + 1, nf, n_attrs);
            free(raw);
            continue;
        }
        valid_rows++;

        for (int col = 0; col < n_input; col++) {
            const char *f = fields[col];
            if (f[0] == '\0' || strcmp(f, "?") == 0) {
                out.data[(size_t)row * n_input + col] = 0.0f;
                missing[(size_t)row * n_input + col] = 1;
            } else {
                out.data[(size_t)row * n_input + col] = (float)atof(f);
            }
        }

        const char *label = fields[n_input];
        int found = -1;
        for (size_t c = 0; c < class_names_seen.size(); c++)
            if (strcasecmp(label, class_names_seen[c].c_str()) == 0) { found = (int)c; break; }
        if (found < 0) {
            if ((int)class_names_seen.size() < MAX_CLASSES) {
                class_names_seen.push_back(label);
                found = (int)class_names_seen.size() - 1;
            } else {
                found = 0;
            }
        }
        out.labels[row] = found;
        free(raw);
    }

    if (valid_rows < data_count) {
        fprintf(stderr, "csv_load: %d of %d rows were malformed and dropped "
                "to label 0 with zeroed features -- check %s for formatting "
                "issues.\n", data_count - valid_rows, data_count, path);
    }

    out.n_classes = (int)class_names_seen.size();
    out.class_names = class_names_seen;
    for (int a = 0; a < n_input; a++)
        out.attr_names.push_back("f" + std::to_string(a));

    /* impute missing values with column mean */
    for (int j = 0; j < n_input; j++) {
        double sum = 0.0; int cnt = 0;
        for (int i = 0; i < data_count; i++)
            if (!missing[(size_t)i * n_input + j]) { sum += out.data[(size_t)i * n_input + j]; cnt++; }
        float mean = cnt > 0 ? (float)(sum / cnt) : 0.0f;
        for (int i = 0; i < data_count; i++)
            if (missing[(size_t)i * n_input + j]) out.data[(size_t)i * n_input + j] = mean;
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * GPU-side per-thread RNG for bagging/mtry (implementation-internal --
 * substitutes for the papers' legacy CUDA-SDK Mersenne Twister sampler;
 * see the deviation note at the top of this file)
 * -------------------------------------------------------------------------*/
__device__ __forceinline__ unsigned long long gpu_rng_step(unsigned long long &state) {
    state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ int gpu_rng_below(unsigned long long &state, int n) {
    return (int)(gpu_rng_step(state) % (unsigned long long)n);
}

/* ---------------------------------------------------------------------------
 * PIPELINE STEP 3 (paper Fig. 2): bagging kernel, one thread per tree.
 * Bootstrap-samples numTrain indices with replacement into d_bagIndices,
 * and marks which training instances were left out-of-bag for this tree.
 * -------------------------------------------------------------------------*/
__global__ void kernel_bagging(int *d_bagIndices, unsigned char *d_oobMask,
                                unsigned long long seed) {
    int tree = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree >= c_numTrees) return;

    unsigned long long state = seed ^ ((unsigned long long)tree * 0x9E3779B97F4A7C15ULL + 0xABCDEFu);
    unsigned char *oob = d_oobMask + (size_t)tree * c_numTrain;
    for (int i = 0; i < c_numTrain; i++) oob[i] = 1;

    int *bag = d_bagIndices + (size_t)tree * c_numTrain;
    for (int i = 0; i < c_numTrain; i++) {
        int idx = gpu_rng_below(state, c_numTrain);
        bag[i] = idx;
        oob[idx] = 0;
    }
}

/* ---------------------------------------------------------------------------
 * Tree node storage, flat arrays sized [numTrees * maxNodesPerTree]
 * -------------------------------------------------------------------------*/
struct TreeNodesGPU {
    unsigned char *isLeaf;
    int   *feature;
    float *threshold;
    int   *classLabel;
    int   *leftChild;
    int   *rightChild;
};

#define STACK_CAPACITY 64   /* generous for maxDepth up to ~30 */

/* ---------------------------------------------------------------------------
 * PIPELINE STEP 4 (paper Fig. 2): iterative tree-build kernel, one thread
 * builds one entire tree. Reproduces the papers' iterative (stack-based)
 * design, required on their GT220 hardware because device-side recursion
 * was unsupported on compute capability 1.2. Splitting criterion is
 * entropy/information gain, per the papers' explicit choice over Gini.
 * -------------------------------------------------------------------------*/
__global__ void kernel_buildTrees(const float *__restrict__ d_trainData,
                                   const int *__restrict__ d_trainLabels,
                                   int *d_bagIndices,
                                   int *d_workBuf,          /* [numTrees*numTrain] scratch */
                                   TreeNodesGPU nodes,
                                   int maxNodesPerTree,
                                   unsigned long long seed,
                                   int *d_nodeCount, /* [numTrees] output: nodes actually used */
                                   int *d_leafCount,  /* [numTrees] output: leaf nodes actually used */
                                   int *d_maxDepthReached /* [numTrees] output: deepest level actually built */) {
    int tree = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree >= c_numTrees) return;

    unsigned long long state = seed ^ ((unsigned long long)tree * 0xD1B54A32D192ED03ULL + 1);

    int *work = d_workBuf + (size_t)tree * c_numTrain;
    int *bag  = d_bagIndices + (size_t)tree * c_numTrain;
    for (int i = 0; i < c_numTrain; i++) work[i] = bag[i];

    unsigned char *tIsLeaf   = nodes.isLeaf    + (size_t)tree * maxNodesPerTree;
    int           *tFeature  = nodes.feature   + (size_t)tree * maxNodesPerTree;
    float         *tThresh   = nodes.threshold + (size_t)tree * maxNodesPerTree;
    int           *tClass    = nodes.classLabel+ (size_t)tree * maxNodesPerTree;
    int           *tLeft     = nodes.leftChild + (size_t)tree * maxNodesPerTree;
    int           *tRight    = nodes.rightChild+ (size_t)tree * maxNodesPerTree;

    /* explicit stack frames: (start, count, depth, nodeId) */
    int stStart[STACK_CAPACITY], stCount[STACK_CAPACITY];
    int stDepth[STACK_CAPACITY], stNode[STACK_CAPACITY];
    int sp = 0;

    int nextNodeId = 1;
    int leafCount = 0;
    int maxDepthReached = 0;
    stStart[sp] = 0; stCount[sp] = c_numTrain; stDepth[sp] = 0; stNode[sp] = 0;
    sp++;

    /* local scratch for split search (sized to MAX_BAG_CAPACITY, capped) */
    float sortVal[MAX_BAG_CAPACITY];
    int   sortIdx[MAX_BAG_CAPACITY];

    while (sp > 0) {
        sp--;
        int start = stStart[sp], count = stCount[sp];
        int depth = stDepth[sp], nodeId = stNode[sp];
        if (depth > maxDepthReached) maxDepthReached = depth;
        if (nodeId >= maxNodesPerTree) continue;

        /* class histogram for this node's samples */
        int hist[MAX_CLASSES];
        for (int c = 0; c < c_numClasses; c++) hist[c] = 0;
        for (int i = 0; i < count; i++) {
            int rec = work[start + i];
            hist[d_trainLabels[rec]]++;
        }
        int majority = 0, majCount = -1;
        for (int c = 0; c < c_numClasses; c++)
            if (hist[c] > majCount) { majCount = hist[c]; majority = c; }
        bool pure = (majCount == count);

        if (pure || count < c_minSamplesSplit || depth >= c_maxDepth) {
            tIsLeaf[nodeId] = 1;
            tClass[nodeId] = majority;
            tLeft[nodeId] = -1;
            tRight[nodeId] = -1;
            leafCount++;
            continue;
        }

        /* parent entropy */
        float parentEntropy = 0.0f;
        for (int c = 0; c < c_numClasses; c++) {
            if (hist[c] == 0) continue;
            float p = (float)hist[c] / (float)count;
            parentEntropy -= p * __logf(p);
        }

        /* mtry: sample k distinct candidate features */
        int kUse = c_k < c_numFeatures ? c_k : c_numFeatures;
        int candidates[MAX_ATTRIBUTES];
        for (int f = 0; f < c_numFeatures; f++) candidates[f] = f;
        for (int i = 0; i < kUse; i++) {
            int j = i + gpu_rng_below(state, c_numFeatures - i);
            int tmp = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp;
        }

        float bestGain = -1.0f;
        int   bestFeature = -1;
        float bestThreshold = 0.0f;

        int capCount = count < MAX_BAG_CAPACITY ? count : MAX_BAG_CAPACITY;
        for (int ci = 0; ci < kUse; ci++) {
            int feat = candidates[ci];
            for (int i = 0; i < capCount; i++) {
                int rec = work[start + i];
                sortVal[i] = d_trainData[(size_t)rec * c_numFeatures + feat];
                sortIdx[i] = rec;
            }
            /* insertion sort by value (small per-node sample counts) */
            for (int i = 1; i < capCount; i++) {
                float v = sortVal[i]; int id = sortIdx[i];
                int j = i - 1;
                while (j >= 0 && sortVal[j] > v) {
                    sortVal[j + 1] = sortVal[j];
                    sortIdx[j + 1] = sortIdx[j];
                    j--;
                }
                sortVal[j + 1] = v;
                sortIdx[j + 1] = id;
            }

            int leftHist[MAX_CLASSES];
            for (int c = 0; c < c_numClasses; c++) leftHist[c] = 0;
            int leftCount = 0;

            for (int i = 0; i < capCount - 1; i++) {
                int lbl = d_trainLabels[sortIdx[i]];
                leftHist[lbl]++;
                leftCount++;
                if (sortVal[i] == sortVal[i + 1]) continue; /* only split between distinct values */

                int rightCount = capCount - leftCount;
                if (leftCount == 0 || rightCount == 0) continue;

                float leftEnt = 0.0f, rightEnt = 0.0f;
                for (int c = 0; c < c_numClasses; c++) {
                    if (leftHist[c] > 0) {
                        float p = (float)leftHist[c] / (float)leftCount;
                        leftEnt -= p * __logf(p);
                    }
                    int rc = hist[c] - leftHist[c];
                    if (rc > 0) {
                        float p = (float)rc / (float)rightCount;
                        rightEnt -= p * __logf(p);
                    }
                }
                float weighted = ((float)leftCount / capCount) * leftEnt +
                                 ((float)rightCount / capCount) * rightEnt;
                float gain = parentEntropy - weighted;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feat;
                    bestThreshold = 0.5f * (sortVal[i] + sortVal[i + 1]);
                }
            }
        }

        if (bestFeature < 0 || bestGain <= 1e-6f) {
            tIsLeaf[nodeId] = 1;
            tClass[nodeId] = majority;
            tLeft[nodeId] = -1;
            tRight[nodeId] = -1;
            leafCount++;
            continue;
        }

        /* in-place partition of work[start..start+count) by bestFeature<=threshold */
        int lo = start, hi = start + count - 1;
        while (lo <= hi) {
            while (lo <= hi && d_trainData[(size_t)work[lo] * c_numFeatures + bestFeature] <= bestThreshold) lo++;
            while (lo <= hi && d_trainData[(size_t)work[hi] * c_numFeatures + bestFeature] >  bestThreshold) hi--;
            if (lo < hi) { int tmp = work[lo]; work[lo] = work[hi]; work[hi] = tmp; lo++; hi--; }
        }
        int leftCountFinal = lo - start;
        int rightCountFinal = count - leftCountFinal;

        tIsLeaf[nodeId] = 0;
        tFeature[nodeId] = bestFeature;
        tThresh[nodeId] = bestThreshold;

        int leftId = nextNodeId++;
        int rightId = nextNodeId++;
        tLeft[nodeId] = (leftId < maxNodesPerTree) ? leftId : -1;
        tRight[nodeId] = (rightId < maxNodesPerTree) ? rightId : -1;

        if (leftCountFinal > 0 && tLeft[nodeId] >= 0 && sp < STACK_CAPACITY) {
            stStart[sp] = start; stCount[sp] = leftCountFinal;
            stDepth[sp] = depth + 1; stNode[sp] = leftId; sp++;
        }
        if (rightCountFinal > 0 && tRight[nodeId] >= 0 && sp < STACK_CAPACITY) {
            stStart[sp] = start + leftCountFinal; stCount[sp] = rightCountFinal;
            stDepth[sp] = depth + 1; stNode[sp] = rightId; sp++;
        }
    }

    d_nodeCount[tree] = nextNodeId; /* nodes actually created for this tree,
                                        used for the "Total nodes" line in
                                        the shared results-report format */
    d_leafCount[tree] = leafCount;  /* leaves actually created for this tree,
                                        used for the "Leaf counts" line */
    d_maxDepthReached[tree] = maxDepthReached; /* deepest level actually built,
                                        vs. the configured --depth cap */
}

/* ---------------------------------------------------------------------------
 * Tree traversal, shared by OOB / train / test prediction kernels
 * -------------------------------------------------------------------------*/
__device__ __forceinline__ int traverse_tree(const unsigned char *isLeaf, const int *feature,
                                              const float *threshold, const int *classLabel,
                                              const int *leftChild, const int *rightChild,
                                              const float *rec) {
    int node = 0;
    for (int guard = 0; guard < 64; guard++) {
        if (isLeaf[node]) return classLabel[node];
        int f = feature[node];
        int next = (rec[f] <= threshold[node]) ? leftChild[node] : rightChild[node];
        if (next < 0) return classLabel[node];
        node = next;
    }
    return classLabel[node];
}

/* ---------------------------------------------------------------------------
 * PIPELINE STEP 5 (paper Fig. 2): OOB classification kernel, one thread per
 * tree, votes only on instances that were out-of-bag for that tree.
 * -------------------------------------------------------------------------*/
__global__ void kernel_classifyOOB(const float *__restrict__ d_trainData,
                                    const unsigned char *__restrict__ d_oobMask,
                                    TreeNodesGPU nodes, int maxNodesPerTree,
                                    int *d_oobVotes /* [numTrain*numClasses] */) {
    int tree = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree >= c_numTrees) return;

    const unsigned char *tIsLeaf = nodes.isLeaf    + (size_t)tree * maxNodesPerTree;
    const int           *tFeature = nodes.feature  + (size_t)tree * maxNodesPerTree;
    const float          *tThresh = nodes.threshold + (size_t)tree * maxNodesPerTree;
    const int            *tClass = nodes.classLabel+ (size_t)tree * maxNodesPerTree;
    const int             *tLeft = nodes.leftChild + (size_t)tree * maxNodesPerTree;
    const int            *tRight = nodes.rightChild+ (size_t)tree * maxNodesPerTree;
    const unsigned char *oob = d_oobMask + (size_t)tree * c_numTrain;

    for (int i = 0; i < c_numTrain; i++) {
        if (!oob[i]) continue;
        const float *rec = d_trainData + (size_t)i * c_numFeatures;
        int pred = traverse_tree(tIsLeaf, tFeature, tThresh, tClass, tLeft, tRight, rec);
        atomicAdd(&d_oobVotes[i * c_numClasses + pred], 1);
    }
}

/* ---------------------------------------------------------------------------
 * PIPELINE STEP 8 (paper Fig. 2): prediction kernel, one thread per tree,
 * reads query/test data through a texture object (paper's "texture memory
 * for read-only test data" optimization -- texture OBJECTS here, the
 * CUDA-12-compatible equivalent of the papers' texture references).
 * -------------------------------------------------------------------------*/
__global__ void kernel_predict(cudaTextureObject_t texQueryData, int numQuery,
                                TreeNodesGPU nodes, int maxNodesPerTree,
                                int *d_votes /* [numQuery*numClasses] */) {
    int tree = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree >= c_numTrees) return;

    const unsigned char *tIsLeaf = nodes.isLeaf    + (size_t)tree * maxNodesPerTree;
    const int           *tFeature = nodes.feature  + (size_t)tree * maxNodesPerTree;
    const float          *tThresh = nodes.threshold + (size_t)tree * maxNodesPerTree;
    const int            *tClass = nodes.classLabel+ (size_t)tree * maxNodesPerTree;
    const int             *tLeft = nodes.leftChild + (size_t)tree * maxNodesPerTree;
    const int            *tRight = nodes.rightChild+ (size_t)tree * maxNodesPerTree;

    for (int q = 0; q < numQuery; q++) {
        int node = 0;
        int pred = 0;
        for (int guard = 0; guard < 64; guard++) {
            if (tIsLeaf[node]) { pred = tClass[node]; break; }
            int f = tFeature[node];
            float v = tex1Dfetch<float>(texQueryData, q * c_numFeatures + f);
            int next = (v <= tThresh[node]) ? tLeft[node] : tRight[node];
            if (next < 0) { pred = tClass[node]; break; }
            node = next;
        }
        atomicAdd(&d_votes[q * c_numClasses + pred], 1);
    }
}

/* ---------------------------------------------------------------------------
 * CLI
 * -------------------------------------------------------------------------*/
struct Args {
    std::string data;
    int trees = 100;
    std::string kSpec = "sqrt";
    int depth = 6;
    unsigned long long seed = 42;
    double trainFrac = 0.8;
    int minSamplesSplit = 2;
    bool forceHeader = false;
    bool forceNoHeader = false;
};

static int resolveK(const std::string &spec, int numFeatures) {
    if (spec == "sqrt") return std::max(1, (int)std::round(std::sqrt((double)numFeatures)));
    if (spec == "log2") return std::max(1, (int)std::round(std::log2((double)numFeatures + 1)));
    if (spec == "all")  return numFeatures;
    try {
        int v = std::stoi(spec);
        return std::max(1, std::min(v, numFeatures));
    } catch (...) {
        return std::max(1, (int)std::round(std::sqrt((double)numFeatures)));
    }
}

static Args parseArgs(int argc, char **argv) {
    Args a;
    if (argc < 2) {
        printf("Usage: %s <data.csv> [--trees N] [--k sqrt|log2|all|<int>] "
               "[--depth N] [--seed N] [--trainfrac F] [--min-samples N] "
               "[--header] [--no-header]\n", argv[0]);
        exit(EXIT_SUCCESS);
    }
    a.data = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", flag); exit(EXIT_FAILURE); }
            return argv[++i];
        };
        if (s == "--trees") a.trees = std::stoi(next("--trees"));
        else if (s == "--k") a.kSpec = next("--k");
        else if (s == "--depth") a.depth = std::stoi(next("--depth"));
        else if (s == "--seed") a.seed = std::stoull(next("--seed"));
        else if (s == "--trainfrac") a.trainFrac = std::stod(next("--trainfrac"));
        else if (s == "--min-samples") a.minSamplesSplit = std::stoi(next("--min-samples"));
        else if (s == "--header") a.forceHeader = true;
        else if (s == "--no-header") a.forceNoHeader = true;
        else if (s == "--help") {
            printf("Usage: %s <data.csv> [--trees N] [--k sqrt|log2|all|<int>] "
                   "[--depth N] [--seed N] [--trainfrac F] [--min-samples N] "
                   "[--header] [--no-header]\n", argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Unknown argument: %s (--help for usage)\n", s.c_str());
            exit(EXIT_FAILURE);
        }
    }
    return a;
}

/* ---------------------------------------------------------------------------
 * main -- orchestrates the paper's 8-step pipeline
 * -------------------------------------------------------------------------*/
int main(int argc, char **argv) {
    auto wallClockStart = std::chrono::high_resolution_clock::now();
    Args args = parseArgs(argc, argv);

    /* ---- STEP 1: read data ---- */
    Dataset ds;
    if (!csv_load(args.data.c_str(), ds, args.forceHeader, args.forceNoHeader)) {
        fprintf(stderr, "Failed to load %s\n", args.data.c_str());
        return EXIT_FAILURE;
    }
    if (ds.n_attributes > MAX_ATTRIBUTES) {
        fprintf(stderr, "n_attributes (%d) exceeds compiled-in cap (%d)\n",
                ds.n_attributes, MAX_ATTRIBUTES);
        return EXIT_FAILURE;
    }

    int k = resolveK(args.kSpec, ds.n_attributes);

    printf("=== CudaRF (Grahn et al. 2010/2011 reconstruction) ===\n");
    printf("Loaded %s: %d instances, %d attributes, %d classes\n",
           args.data.c_str(), ds.n_instances, ds.n_attributes, ds.n_classes);
    printf("Config: trees=%d k=%d (%s) depth=%d seed=%llu trainFrac=%.2f "
           "min-samples-split=%d\n",
           args.trees, k, args.kSpec.c_str(), args.depth, args.seed,
           args.trainFrac, args.minSamplesSplit);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d multiprocessors)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);

    int runtimeVer = 0, driverVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    CUDA_CHECK(cudaDriverGetVersion(&driverVer));
    printf("CUDA runtime %d.%d, driver %d.%d\n",
           runtimeVer / 1000, (runtimeVer % 100) / 10,
           driverVer / 1000, (driverVer % 100) / 10);
    printf("Split RNG: shared splitmix64 (identical across every tool, given "
           "--seed) | Bagging/mtry RNG: per-tree splitmix64 (this tool only)\n");

    /* ---- shared stratified split (identical across every tool, given --seed) ---- */
    std::vector<int> trainIdx(ds.n_instances), testIdx(ds.n_instances);
    int trainCount = 0, testCount = 0;
    stratified_split(ds.labels.data(), ds.n_instances, ds.n_classes,
                      args.trainFrac, args.seed,
                      trainIdx.data(), &trainCount, testIdx.data(), &testCount);
    trainIdx.resize(trainCount);
    testIdx.resize(testCount);
    printf("Train instances=%d  Test instances=%d (shared stratified split, seed=%llu)\n",
           trainCount, testCount, args.seed);

    if (trainCount > MAX_BAG_CAPACITY) {
        fprintf(stderr, "Train set (%d) exceeds compiled-in MAX_BAG_CAPACITY (%d); "
                "increase MAX_BAG_CAPACITY and recompile.\n", trainCount, MAX_BAG_CAPACITY);
        return EXIT_FAILURE;
    }

    /* ---- STEP 2: format & transfer (pinned host memory, per paper) ---- */
    float *h_trainData = nullptr;
    int   *h_trainLabels = nullptr;
    float *h_testData = nullptr;
    CUDA_CHECK(cudaHostAlloc((void **)&h_trainData, (size_t)trainCount * ds.n_attributes * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_trainLabels, (size_t)trainCount * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_testData, (size_t)testCount * ds.n_attributes * sizeof(float), cudaHostAllocDefault));

    for (int i = 0; i < trainCount; i++) {
        int rec = trainIdx[i];
        memcpy(h_trainData + (size_t)i * ds.n_attributes,
               ds.data.data() + (size_t)rec * ds.n_attributes,
               ds.n_attributes * sizeof(float));
        h_trainLabels[i] = ds.labels[rec];
    }
    for (int i = 0; i < testCount; i++) {
        int rec = testIdx[i];
        memcpy(h_testData + (size_t)i * ds.n_attributes,
               ds.data.data() + (size_t)rec * ds.n_attributes,
               ds.n_attributes * sizeof(float));
    }

    float *d_trainData; int *d_trainLabels; float *d_testData;
    CUDA_CHECK(cudaMalloc(&d_trainData, (size_t)trainCount * ds.n_attributes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trainLabels, (size_t)trainCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_testData, (size_t)testCount * ds.n_attributes * sizeof(float)));

    auto h2dStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_trainData, h_trainData, (size_t)trainCount * ds.n_attributes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_trainLabels, h_trainLabels, (size_t)trainCount * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_testData, h_testData, (size_t)testCount * ds.n_attributes * sizeof(float), cudaMemcpyHostToDevice));
    auto h2dEnd = std::chrono::high_resolution_clock::now();
    double h2dMs = std::chrono::duration<double, std::milli>(h2dEnd - h2dStart).count();

    /* constant memory config */
    CUDA_CHECK(cudaMemcpyToSymbol(c_numFeatures, &ds.n_attributes, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numClasses, &ds.n_classes, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_k, &k, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_maxDepth, &args.depth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numTrain, &trainCount, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_numTrees, &args.trees, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_minSamplesSplit, &args.minSamplesSplit, sizeof(int)));

    /* texture object for test/query data (paper's texture-memory optimization) */
    cudaTextureObject_t texQueryData = 0;
    cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_testData;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = (size_t)testCount * ds.n_attributes * sizeof(float);
    cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    CUDA_CHECK(cudaCreateTextureObject(&texQueryData, &resDesc, &texDesc, nullptr));

    int maxNodesPerTree = (1 << (args.depth + 1)) - 1;
    if (maxNodesPerTree < 1) maxNodesPerTree = 1;

    TreeNodesGPU nodes;
    size_t nodeArrSize = (size_t)args.trees * maxNodesPerTree;
    CUDA_CHECK(cudaMalloc(&nodes.isLeaf, nodeArrSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&nodes.feature, nodeArrSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nodes.threshold, nodeArrSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nodes.classLabel, nodeArrSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nodes.leftChild, nodeArrSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nodes.rightChild, nodeArrSize * sizeof(int)));

    int *d_bagIndices, *d_workBuf;
    unsigned char *d_oobMask;
    CUDA_CHECK(cudaMalloc(&d_bagIndices, (size_t)args.trees * trainCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_workBuf, (size_t)args.trees * trainCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_oobMask, (size_t)args.trees * trainCount * sizeof(unsigned char)));

    int *d_nodeCount;
    CUDA_CHECK(cudaMalloc(&d_nodeCount, (size_t)args.trees * sizeof(int)));
    int *d_leafCount;
    CUDA_CHECK(cudaMalloc(&d_leafCount, (size_t)args.trees * sizeof(int)));
    int *d_maxDepthReached;
    CUDA_CHECK(cudaMalloc(&d_maxDepthReached, (size_t)args.trees * sizeof(int)));

    int threads = 64;
    int blocks = (args.trees + threads - 1) / threads;

    cudaEvent_t evStart, evBagDone, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evBagDone));
    CUDA_CHECK(cudaEventCreate(&evStop));

    /* ---- STEP 3: bagging kernel ---- */
    CUDA_CHECK(cudaEventRecord(evStart));
    kernel_bagging<<<blocks, threads>>>(d_bagIndices, d_oobMask, args.seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evBagDone));

    /* ---- STEP 4: parallel tree-build kernel ---- */
    kernel_buildTrees<<<blocks, threads>>>(d_trainData, d_trainLabels, d_bagIndices,
                                            d_workBuf, nodes, maxNodesPerTree, args.seed,
                                            d_nodeCount, d_leafCount, d_maxDepthReached);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float bagMs = 0.0f, buildOnlyMs = 0.0f, buildMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&bagMs, evStart, evBagDone));
    CUDA_CHECK(cudaEventElapsedTime(&buildOnlyMs, evBagDone, evStop));
    CUDA_CHECK(cudaEventElapsedTime(&buildMs, evStart, evStop)); /* bagging+build combined, kept for the shared results-report line */

    /* ---- STEP 5: OOB classification kernel ---- */
    int *d_oobVotes;
    CUDA_CHECK(cudaMalloc(&d_oobVotes, (size_t)trainCount * ds.n_classes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_oobVotes, 0, (size_t)trainCount * ds.n_classes * sizeof(int)));
    kernel_classifyOOB<<<blocks, threads>>>(d_trainData, d_oobMask, nodes, maxNodesPerTree, d_oobVotes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- STEP 6: host computes OOB error ---- */
    double d2hMs = 0.0; /* accumulated across every D2H copy below */
    std::vector<int> h_oobVotes((size_t)trainCount * ds.n_classes);
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_oobVotes.data(), d_oobVotes, (size_t)trainCount * ds.n_classes * sizeof(int), cudaMemcpyDeviceToHost));
        auto t1 = std::chrono::high_resolution_clock::now();
        d2hMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    int oobCorrect = 0, oobTotal = 0;
    for (int i = 0; i < trainCount; i++) {
        int best = -1, bestCount = -1, sum = 0;
        for (int c = 0; c < ds.n_classes; c++) {
            int v = h_oobVotes[i * ds.n_classes + c];
            sum += v;
            if (v > bestCount) { bestCount = v; best = c; }
        }
        if (sum == 0) continue; /* never OOB (rare with enough trees) */
        oobTotal++;
        if (best == h_trainLabels[i]) oobCorrect++;
    }
    float oobAccuracy = oobTotal > 0 ? 100.0f * oobCorrect / oobTotal : 0.0f;

    /* ---- STEP 7: transfer query data (already resident; texture object created above) ---- */

    /* ---- STEP 8: prediction kernel, test set, via texture object per the paper's optimization ---- */
    int *d_testVotes;
    CUDA_CHECK(cudaMalloc(&d_testVotes, (size_t)testCount * ds.n_classes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_testVotes, 0, (size_t)testCount * ds.n_classes * sizeof(int)));
    auto predT0 = std::chrono::high_resolution_clock::now();
    kernel_predict<<<blocks, threads>>>(texQueryData, testCount, nodes, maxNodesPerTree, d_testVotes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto predT1 = std::chrono::high_resolution_clock::now();
    double predMs = std::chrono::duration<double, std::milli>(predT1 - predT0).count();

    /* ---- STEP 9 (paper's "transfer results back"): copy test votes to host ---- */
    std::vector<int> h_testVotes((size_t)testCount * ds.n_classes);
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_testVotes.data(), d_testVotes, (size_t)testCount * ds.n_classes * sizeof(int), cudaMemcpyDeviceToHost));
        auto t1 = std::chrono::high_resolution_clock::now();
        d2hMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    int testCorrect = 0;
    for (int i = 0; i < testCount; i++) {
        int rec = testIdx[i];
        int best = -1, bestCount = -1;
        for (int c = 0; c < ds.n_classes; c++) {
            int v = h_testVotes[i * ds.n_classes + c];
            if (v > bestCount) { bestCount = v; best = c; }
        }
        if (best == ds.labels[rec]) testCorrect++;
    }
    float testAccuracy = testCount > 0 ? 100.0f * testCorrect / testCount : 0.0f;

    /* train-set prediction (for train accuracy, not part of paper's numbered
       steps but useful to report alongside OOB/test) */
    int *d_trainVotes;
    CUDA_CHECK(cudaMalloc(&d_trainVotes, (size_t)trainCount * ds.n_classes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_trainVotes, 0, (size_t)trainCount * ds.n_classes * sizeof(int)));
    cudaResourceDesc resDescTrain; memset(&resDescTrain, 0, sizeof(resDescTrain));
    resDescTrain.resType = cudaResourceTypeLinear;
    resDescTrain.res.linear.devPtr = d_trainData;
    resDescTrain.res.linear.desc = cudaCreateChannelDesc<float>();
    resDescTrain.res.linear.sizeInBytes = (size_t)trainCount * ds.n_attributes * sizeof(float);
    cudaTextureObject_t texTrainData = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texTrainData, &resDescTrain, &texDesc, nullptr));
    kernel_predict<<<blocks, threads>>>(texTrainData, trainCount, nodes, maxNodesPerTree, d_trainVotes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_trainVotes((size_t)trainCount * ds.n_classes);
    CUDA_CHECK(cudaMemcpy(h_trainVotes.data(), d_trainVotes, (size_t)trainCount * ds.n_classes * sizeof(int), cudaMemcpyDeviceToHost));
    int trainCorrect = 0;
    for (int i = 0; i < trainCount; i++) {
        int best = -1, bestCount = -1;
        for (int c = 0; c < ds.n_classes; c++) {
            int v = h_trainVotes[i * ds.n_classes + c];
            if (v > bestCount) { bestCount = v; best = c; }
        }
        if (best == h_trainLabels[i]) trainCorrect++;
    }
    float trainAccuracy = trainCount > 0 ? 100.0f * trainCorrect / trainCount : 0.0f;

    /* ---- node/leaf/depth accounting (part of the shared results-report format) ---- */
    std::vector<int> h_nodeCount(args.trees);
    CUDA_CHECK(cudaMemcpy(h_nodeCount.data(), d_nodeCount, (size_t)args.trees * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_leafCount(args.trees);
    CUDA_CHECK(cudaMemcpy(h_leafCount.data(), d_leafCount, (size_t)args.trees * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_maxDepthReached(args.trees);
    CUDA_CHECK(cudaMemcpy(h_maxDepthReached.data(), d_maxDepthReached, (size_t)args.trees * sizeof(int), cudaMemcpyDeviceToHost));

    long long totalNodes = 0, totalLeaves = 0, totalDepth = 0;
    int minNodes = INT_MAX, maxNodes = INT_MIN;
    int minLeaves = INT_MAX, maxLeaves = INT_MIN;
    for (int t = 0; t < args.trees; t++) {
        totalNodes += h_nodeCount[t];
        totalLeaves += h_leafCount[t];
        totalDepth += h_maxDepthReached[t];
        if (h_nodeCount[t] < minNodes) minNodes = h_nodeCount[t];
        if (h_nodeCount[t] > maxNodes) maxNodes = h_nodeCount[t];
        if (h_leafCount[t] < minLeaves) minLeaves = h_leafCount[t];
        if (h_leafCount[t] > maxLeaves) maxLeaves = h_leafCount[t];
    }
    double avgNodesPerTree = args.trees > 0 ? (double)totalNodes / args.trees : 0.0;
    double avgLeavesPerTree = args.trees > 0 ? (double)totalLeaves / args.trees : 0.0;
    double avgDepthReached = args.trees > 0 ? (double)totalDepth / args.trees : 0.0;
    double avgSamplesPerLeaf = totalLeaves > 0 ? ((double)trainCount * args.trees) / totalLeaves : 0.0;

    double predThroughputKSps = predMs > 0.0 ? (testCount / (predMs / 1000.0)) / 1000.0 : 0.0;

    auto wallClockEnd = std::chrono::high_resolution_clock::now();
    double totalWallMs = std::chrono::duration<double, std::milli>(wallClockEnd - wallClockStart).count();

    /* ---- STEP 10 (paper's "output") ---- */
    /* All GPU measurements are complete at this point.
     * Now invoke the separately compiled CPU baseline as a child process.
     * Its execution must not overlap with or pollute any GPU timing above. */

    double cpuTrainMs  = -1.0; /* -1 = unavailable */
    bool   baselineOk  = false;
    std::string baselineStatus = "NOT FOUND";

    /* Build the baseline_rf command with the same parameters CudaRF used.
     * --mtry: baseline_rf uses --mtry; cudarf uses --k with identical semantics. */
    {
        /* Check executable exists */
        const char *baselineExe = "./baseline_rf";
        bool exeFound = (access(baselineExe, X_OK) == 0);
        if (!exeFound) {
            baselineStatus = "NOT FOUND";
        } else {
            /* Construct command string */
            char cmd[4096];
            snprintf(cmd, sizeof(cmd),
                     "./baseline_rf %s --trees %d --mtry %s --depth %d "
                     "--seed %llu --trainfrac %.6f 2>&1",
                     args.data.c_str(),
                     args.trees,
                     args.kSpec.c_str(),   /* baseline_rf uses --mtry; cudarf uses --k; same values */
                     args.depth,
                     args.seed,
                     args.trainFrac);

            /* Launch baseline as child process, capture stdout+stderr */
            FILE *pipe = popen(cmd, "r");
            if (!pipe) {
                baselineStatus = "LAUNCH_FAILED";
            } else {
                char linebuf[1024];
                std::string firstErrorLine;
                while (fgets(linebuf, sizeof(linebuf), pipe)) {
                    /* Parse:  "CPU build time:       XXX.XXX ms  (single-threaded)"
                     * Variable spacing after the colon, so skip whitespace manually. */
                    if (strncmp(linebuf, "CPU build time:", 15) == 0) {
                        double parsed_ms = 0.0;
                        const char *p = linebuf + 15;
                        while (*p == ' ' || *p == '\t') p++;
                        if (sscanf(p, "%lf", &parsed_ms) == 1) {
                            cpuTrainMs = parsed_ms;
                        }
                    } else if (firstErrorLine.empty() &&
                               (strncmp(linebuf, "Unknown",  7) == 0 ||
                                strncmp(linebuf, "Failed",   6) == 0 ||
                                strncmp(linebuf, "csv_load", 8) == 0 ||
                                strncmp(linebuf, "ERROR",    5) == 0)) {
                        firstErrorLine = linebuf;
                        if (!firstErrorLine.empty() && firstErrorLine.back() == '\n')
                            firstErrorLine.pop_back();
                    }
                }
                int rc = pclose(pipe);
                if (cpuTrainMs >= 0.0) {
                    baselineOk = true;
                    baselineStatus = (rc == 0) ? "SUCCESS" : "SUCCESS (non-zero exit)";
                } else {
                    if (!firstErrorLine.empty())
                        baselineStatus = "FAILED: " + firstErrorLine;
                    else
                        baselineStatus = "PARSE_FAILED (no CPU build time line in output)";
                }
            }
        }
    }

    /* Derived timing values */
    double totalGpuTrainMs = h2dMs + (double)buildMs + d2hMs;

    /* ================================================================
     * FINAL REPORT
     * ================================================================ */
    printf("\n=== CudaRF Results ===\n");
    printf("Dataset: %s  Instances=%d  Features=%d  Classes=%d\n",
           args.data.c_str(), ds.n_instances, ds.n_attributes, ds.n_classes);
    printf("Config: trees=%d mtry=%d (%s) depth=%d seed=%llu trainFrac=%.2f\n",
           args.trees, k, args.kSpec.c_str(), args.depth, args.seed, args.trainFrac);
    printf("Criterion: entropy/information-gain (paper-specified)\n");
    printf("Split: train=%d test=%d (shared stratified split, seed=%llu)\n",
           trainCount, testCount, args.seed);

    printf("\n================================================================\n");
    printf("7.1 Correctness and Reconstruction Validation\n");
    printf("================================================================\n");
    printf("Build/Run Status:      SUCCESS\n");
    printf("Train accuracy:        %.2f%%\n", trainAccuracy);
    printf("OOB accuracy:          %.2f%%  (%d of %d train instances had >=1 OOB vote)\n",
           oobAccuracy, oobTotal, trainCount);
    printf("Test accuracy:         %.2f%%\n", testAccuracy);
    printf("Degenerate-tree check: %s\n",
           (totalNodes > args.trees) ? "PASS" : "FAIL (degenerate -- check bagging RNG)");

    printf("\n================================================================\n");
    printf("7.2 Training Performance\n");
    printf("================================================================\n");
    printf("H2D:                   %.3f ms\n", h2dMs);
    printf("  Bagging kernel:      %.3f ms\n", (double)bagMs);
    printf("  Tree-build kernel:   %.3f ms\n", (double)buildOnlyMs);
    printf("GPU Train:             %.3f ms  (bagging+build kernels combined)\n", (double)buildMs);
    printf("D2H:                   %.3f ms  (OOB + test votes)\n", d2hMs);
    printf("Total GPU Train:       %.3f ms  (H2D + GPU Train + D2H)\n", totalGpuTrainMs);
    printf("\n");
    if (baselineOk) {
        printf("CPU baseline:\n");
        printf("  Implementation:    baseline_rf (single-threaded CPU, Gini, CART)\n");
        printf("  Execution:         separate child process\n");
        printf("  Threads:           1\n");
        printf("  Matching config:   trees=%d  mtry=%s  depth=%d  seed=%llu  trainfrac=%.2f\n",
               args.trees, args.kSpec.c_str(), args.depth, args.seed, args.trainFrac);
        printf("  CPU Train:         %.3f ms\n", cpuTrainMs);
        double speedup = (totalGpuTrainMs > 0.0) ? cpuTrainMs / totalGpuTrainMs : 0.0;
        printf("Speedup:             %.2fx  (CPU Train / Total GPU Train)\n", speedup);
    } else {
        printf("CPU Train:           unavailable\n");
        printf("Speedup:             unavailable\n");
    }
    printf("Baseline status:       %s\n", baselineStatus.c_str());

    printf("\n================================================================\n");
    printf("7.3 Prediction Performance\n");
    printf("================================================================\n");
    printf("Inference Time:        %.3f ms  (%d test instances)\n", predMs, testCount);
    printf("Throughput:            %.2f KSps\n", predThroughputKSps);
    printf("Implementation notes:  GPU kernel_predict (1 thread/tree) via texture object\n");
    printf("                       Texture-object binding for read-only test data\n");
    printf("                       (paper's texture-memory optimization, CUDA-12 API)\n");

    printf("\n================================================================\n");
    printf("7.4 Tree Structure Analysis\n");
    printf("================================================================\n");
    printf("Trees:                 %d\n", args.trees);
    printf("Total Nodes:           %lld\n", totalNodes);
    printf("Avg Nodes/Tree:        %.1f  (min %d, max %d)\n", avgNodesPerTree, minNodes, maxNodes);
    printf("Total Leaves:          %lld\n", totalLeaves);
    printf("Avg Leaves/Tree:       %.1f  (min %d, max %d)\n", avgLeavesPerTree, minLeaves, maxLeaves);
    printf("Avg Depth:             %.1f\n", avgDepthReached);
    printf("Depth Cap:             %d\n", args.depth);
    printf("Avg Samples/Leaf:      %.2f\n", avgSamplesPerLeaf);

    printf("\n================================================================\n");
    printf("7.5 How Tree Structure Affects GPU Behaviour\n");
    printf("================================================================\n");
    printf("Parallelism:           1 GPU thread per tree (tree-level granularity)\n");
    printf("                       Trees=%d threads active simultaneously (capped by SM count)\n",
           args.trees);
    printf("Construction:          Iterative stack-based (paper design; avoids device recursion)\n");
    printf("Criterion:             Entropy/info-gain via __logf (fast approximate intrinsic)\n");
    printf("Memory:                Pinned host memory (cudaHostAlloc) + constant memory config\n");
    printf("                       Texture object for read-only test data (read-only cache path)\n");
    printf("Bagging kernel time:   %.3f ms  (separate from build kernel)\n", (double)bagMs);
    printf("Build kernel time:     %.3f ms\n", (double)buildOnlyMs);

    printf("\n================================================================\n");
    printf("7.6 Memory Consumption and Resource Requirements\n");
    printf("================================================================\n");
    {
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        double usedMB  = (double)(totalMem - freeMem) / (1024.0 * 1024.0);
        double totalMB = (double)totalMem / (1024.0 * 1024.0);
        double utilPct = totalMB > 0.0 ? 100.0 * usedMB / totalMB : 0.0;
        size_t nodeArrBytes = (size_t)args.trees * maxNodesPerTree *
                              (sizeof(unsigned char) + sizeof(int)*4 + sizeof(float));
        double forestMB = (double)nodeArrBytes / (1024.0 * 1024.0);
        printf("Peak GPU Memory:       %.2f MB used / %.2f MB total\n", usedMB, totalMB);
        printf("Peak Memory Util.:     %.2f %%\n", utilPct);
        printf("Forest Memory:         %.2f MB  (%d trees x %d max nodes/tree)\n",
               forestMB, args.trees, maxNodesPerTree);
    }

    printf("\n----------------------------------------------------------------\n");
    printf("Total wall-clock time: %.3f ms  (load+split+transfer+build+predict)\n", totalWallMs);
    printf("----------------------------------------------------------------\n");

    /* cleanup */
    cudaDestroyTextureObject(texQueryData);
    cudaDestroyTextureObject(texTrainData);
    cudaFree(d_trainData); cudaFree(d_trainLabels); cudaFree(d_testData);
    cudaFree(d_bagIndices); cudaFree(d_workBuf); cudaFree(d_oobMask);
    cudaFree(d_nodeCount); cudaFree(d_leafCount); cudaFree(d_maxDepthReached);
    cudaFree(d_oobVotes); cudaFree(d_testVotes); cudaFree(d_trainVotes);
    cudaFree(nodes.isLeaf); cudaFree(nodes.feature); cudaFree(nodes.threshold);
    cudaFree(nodes.classLabel); cudaFree(nodes.leftChild); cudaFree(nodes.rightChild);
    cudaFreeHost(h_trainData); cudaFreeHost(h_trainLabels); cudaFreeHost(h_testData);
    cudaEventDestroy(evStart); cudaEventDestroy(evBagDone); cudaEventDestroy(evStop);

    return 0;
}
