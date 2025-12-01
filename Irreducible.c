#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <strings.h>   /* for strcasecmp on Linux */

#include <ilcplex/cplex.h>

/* ---------- Utilities ---------- */

static void die(const char *msg) {
    fprintf(stderr, "[FATAL] %s\n", msg);
    exit(EXIT_FAILURE);
}

static void die_perror(const char *msg, const char *path) {
    fprintf(stderr, "[FATAL] %s '%s': %s\n", msg, path, strerror(errno));
    exit(EXIT_FAILURE);
}

static void log_info(const char *msg) {
    fprintf(stdout, "[LOG] %s\n", msg);
}

/* ---------- Simple robust CSV splitting (handles quoted fields) ---------- */

static void strip_utf8_bom(char *line) {
    unsigned char *u = (unsigned char*)line;
    if (u[0] == 0xEF && u[1] == 0xBB && u[2] == 0xBF) {
        memmove(line, line + 3, strlen(line + 3) + 1);
    }
}

static char detect_delim(const char *line) {
    int commas = 0, semis = 0, tabs = 0;
    for (const char *p = line; *p; ++p) {
        if (*p == ',') commas++;
        else if (*p == ';') semis++;
        else if (*p == '\t') tabs++;
    }
    if (semis > commas && semis >= tabs) return ';';
    if (tabs  > commas && tabs  >  semis) return '\t';
    return ',';
}

/* Splits line in-place; returns number of tokens, tokens[] pointers into line */
static int split_csv_quoted_delim(char *line, char **tokens, int max_tokens, char delim) {
    int nt = 0;
    char *p = line;

    while (*p && nt < max_tokens) {
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '\0' || *p == '\r' || *p == '\n') break;

        if (*p == '"') {
            p++;
            char *start = p;
            char *out = p;

            while (*p) {
                if (*p == '"' && p[1] == '"') {
                    *out++ = '"';
                    p += 2;
                    continue;
                }
                if (*p == '"') {
                    p++;
                    break;
                }
                *out++ = *p++;
            }
            *out = '\0';
            tokens[nt++] = start;

            while (*p && *p != delim && *p != '\r' && *p != '\n') p++;
            if (*p == delim) p++;
            while (*p == '\r' || *p == '\n') p++;
            continue;
        }

        char *start = p;
        while (*p && *p != delim && *p != '\r' && *p != '\n') p++;

        if (*p == delim) {
            *p = '\0';
            p++;
        } else if (*p == '\r' || *p == '\n') {
            *p = '\0';
            while (*p == '\r' || *p == '\n') p++;
        }

        tokens[nt++] = start;
    }

    return nt;
}

static char *trim(char *s) {
    if (!s) return s;
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) --e;
    *e = '\0';
    return s;
}

static void normalize_code_inplace(char *s) {
    s = trim(s);
    size_t n = strlen(s);
    if (n == 0) return;

    if (s[0] == '"' && n >= 2 && s[n-1] == '"') {
        memmove(s, s+1, n-2);
        s[n-2] = '\0';
        n -= 2;
    }

    char *dot = strchr(s, '.');
    if (dot) {
        int only_zeros = 1;
        for (char *p = dot+1; *p; ++p) {
            if (*p != '0') { only_zeros = 0; break; }
        }
        if (only_zeros) *dot = '\0';
    }
}

/* ---------- RNG ---------- */

typedef struct { uint64_t state; } rng64_t;

static uint64_t rng_next_u64(rng64_t *r) {
    uint64_t z = (r->state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t uniform_u64_bounded(rng64_t *r, uint64_t bound) {
    if (bound == 0) return 0;
    uint64_t x, lim = UINT64_MAX - (UINT64_MAX % bound);
    do { x = rng_next_u64(r); } while (x >= lim);
    return x % bound;
}

/* ---------- Hash set for uint64 (open addressing) ---------- */

typedef struct {
    uint64_t *keys;
    uint8_t  *used;
    size_t cap;
    size_t n;
} u64set_t;

static size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void u64set_init(u64set_t *s, size_t expected) {
    s->cap = next_pow2(expected * 2 + 16);
    s->keys = (uint64_t*)malloc(s->cap * sizeof(uint64_t));
    s->used = (uint8_t*)calloc(s->cap, sizeof(uint8_t));
    if (!s->keys || !s->used) die("Out of memory initializing hash set");
    s->n = 0;
}

static void u64set_free(u64set_t *s) {
    free(s->keys);
    free(s->used);
    s->keys = NULL; s->used = NULL; s->cap = 0; s->n = 0;
}

static int u64set_has(const u64set_t *s, uint64_t key) {
    size_t mask = s->cap - 1;
    size_t i = (size_t)hash_u64(key) & mask;
    while (s->used[i]) {
        if (s->keys[i] == key) return 1;
        i = (i + 1) & mask;
    }
    return 0;
}

static void u64set_add(u64set_t *s, uint64_t key) {
    if (s->n * 10 >= s->cap * 7) die("Hash set too full (increase expected size)");
    size_t mask = s->cap - 1;
    size_t i = (size_t)hash_u64(key) & mask;
    while (s->used[i]) {
        if (s->keys[i] == key) return;
        i = (i + 1) & mask;
    }
    s->used[i] = 1;
    s->keys[i] = key;
    s->n++;
}

/* ---------- Data structures ---------- */

typedef struct {
    int    S;
    char **labels;  /* sector labels [0..S-1] as normalized strings */
    double *Irow;   /* incidence-like I (buyer rows, seller cols), size S*S */
} IOTData;

typedef struct {
    int N;
    double *q;
    int *sector;
} FirmData;

typedef struct {
    int n;
    int m;
    int *src;
    int *dst;
} EdgeList;

static inline uint64_t edge_key(int u, int v){
    return ((uint64_t)(uint32_t)u << 32) | (uint32_t)v;
}

static void build_edge_set(const EdgeList *el, u64set_t *set){
    u64set_init(set, (size_t)el->m + 16);
    for(int e=0;e<el->m;e++){
        int u = el->src[e], v = el->dst[e];
        if(u==v) continue;
        u64set_add(set, edge_key(u,v));
    }
}

typedef struct {
    int n, m;
    int *head;
    int *to;
    int *next;
} Adj;

/* ---------- Read IO CSV & build Irow ---------- */

static void read_iot_csv(const char *path, IOTData *iot) {
    FILE *fp = fopen(path, "r");
    if (!fp) die_perror("Cannot open IO CSV", path);

    log_info("Reading IO table and building sector incidence matrix...");
    char line[200000];
    char raw[200000];
    char *tok[2048];

    char delim = ',';
    int nt = 0;

    int tries = 0;
    while (fgets(line, sizeof(line), fp)) {
        tries++;
        strncpy(raw, line, sizeof(raw)-1);
        raw[sizeof(raw)-1] = '\0';

        strip_utf8_bom(line);
        char *t = trim(line);
        if (*t == '\0') continue;

        if (!strncasecmp(t, "sep=", 4) && t[4] != '\0') {
            delim = t[4];
            continue;
        }

        delim = detect_delim(t);
        nt = split_csv_quoted_delim(t, tok, 2048, delim);

        if (nt >= 2) break;
        if (tries > 50) break;
    }

    if (nt < 2) {
        fclose(fp);
        fprintf(stderr,
            "[FATAL] IOT header malformed. Couldn't find a delimited header (>=2 cols).\n"
            "        Detected delimiter guess='%c'.\n"
            "        First raw line was:\n%s\n",
            delim, raw);
        exit(EXIT_FAILURE);
    }

    int S = nt - 1;
    iot->S = S;
    iot->labels = (char**)calloc(S, sizeof(char*));
    if (!iot->labels) die("Out of memory for IO labels");

    for (int j = 0; j < S; ++j) {
        char *lab = trim(tok[j+1]);
        normalize_code_inplace(lab);
        size_t len = strlen(lab);
        iot->labels[j] = (char*)malloc(len + 1);
        if (!iot->labels[j]) die("Out of memory for IO label dup");
        memcpy(iot->labels[j], lab, len + 1);
    }

    double *IO = (double*)calloc((size_t)S * (size_t)S, sizeof(double));
    if (!IO) die("Out of memory for IO matrix");

    int row = 0;
    while (row < S && fgets(line, sizeof(line), fp)) {
        strip_utf8_bom(line);
        char *t = trim(line);
        if (*t == '\0') continue;

        int nr = split_csv_quoted_delim(t, tok, 2048, delim);
        if (nr < S + 1) die("IOT row has fewer columns than header");

        for (int j = 0; j < S; ++j) {
            char *s = trim(tok[j+1]);
            double v = 0.0;
            if (*s) v = atof(s);
            if (!isfinite(v) || v < 0.0) v = 0.0;
            IO[row * S + j] = v;
        }
        row++;
    }
    fclose(fp);

    if (row != S) {
        fprintf(stderr, "[WARN] IOT had %d rows, expected %d. Truncating to %d.\n", row, S, row);
        iot->S = row;
        S = row;
    }

    iot->Irow = (double*)calloc((size_t)S * (size_t)S, sizeof(double));
    if (!iot->Irow) die("Out of memory for Irow");

    for (int k = 0; k < S; ++k) {
        for (int l = 0; l < S; ++l) {
            iot->Irow[k * S + l] = (IO[k * S + l] > 0.0) ? 1.0 : 0.0;
        }
    }

    free(IO);
    fprintf(stdout, "[LOG] IO loaded: S=%d sectors (incidence mode).\n", iot->S);
}

/* ---------- Read firm strength + mapping (row-aligned) ---------- */

typedef struct {
    int N;
    char **firm_id;
    double *size;
} FirmStrengthRaw;

typedef struct {
    int N;
    char **firm_id;
    char **sic;
} MappingRaw;

static void read_firm_strength_csv(const char *path, FirmStrengthRaw *fr) {
    FILE *fp = fopen(path, "r");
    if (!fp) die_perror("Cannot open firm strength CSV", path);

    log_info("Reading firm strength CSV...");
    char line[200000];
    char *tok[2048];

    if (!fgets(line, sizeof(line), fp)) die("Firm strength CSV empty");
    int nt = split_csv_quoted_delim(line, tok, 2048, ',');

    int idx_firm = -1, idx_size = -1;
    for (int i = 0; i < nt; ++i) {
        char *h = trim(tok[i]);
        if (!strcasecmp(h, "Firm_ID") || !strcasecmp(h, "FirmID")) idx_firm = i;
        if (!strcasecmp(h, "Size")) idx_size = i;
    }
    if (idx_firm < 0 || idx_size < 0) die("Firm strength CSV must have Firm_ID and Size columns");

    long pos = ftell(fp);
    int N = 0;
    while (fgets(line, sizeof(line), fp)) N++;
    if (N <= 0) die("Firm strength CSV has no data rows");
    fseek(fp, pos, SEEK_SET);

    fr->N = N;
    fr->firm_id = (char**)calloc(N, sizeof(char*));
    fr->size = (double*)calloc(N, sizeof(double));
    if (!fr->firm_id || !fr->size) die("Out of memory for firm strength arrays");

    int row = 0;
    while (row < N && fgets(line, sizeof(line), fp)) {
        int nr = split_csv_quoted_delim(line, tok, 2048, ',');
        if (nr <= idx_firm || nr <= idx_size) die("Firm strength row too short");

        char *fid = trim(tok[idx_firm]);
        normalize_code_inplace(fid);
        size_t len = strlen(fid);
        fr->firm_id[row] = (char*)malloc(len + 1);
        if (!fr->firm_id[row]) die("Out of memory dup firm_id");
        memcpy(fr->firm_id[row], fid, len + 1);

        char *sz = trim(tok[idx_size]);
        double v = (*sz) ? atof(sz) : 0.0;
        if (!isfinite(v) || v < 0) v = 0.0;
        fr->size[row] = v;

        row++;
    }
    fclose(fp);
    fprintf(stdout, "[LOG] Firm strength loaded: N=%d\n", fr->N);
}

static void read_mapping_csv(const char *path, MappingRaw *mr) {
    FILE *fp = fopen(path, "r");
    if (!fp) die_perror("Cannot open mapping CSV", path);

    log_info("Reading mapping CSV...");
    char line[200000];
    char *tok[2048];

    if (!fgets(line, sizeof(line), fp)) die("Mapping CSV empty");
    int nt = split_csv_quoted_delim(line, tok, 2048, ',');

    int idx_firm = -1, idx_sic = -1;
    for (int i = 0; i < nt; ++i) {
        char *h = trim(tok[i]);
        if (!strcasecmp(h, "Firm_ID") || !strcasecmp(h, "FirmID")) idx_firm = i;
        if (!strcasecmp(h, "SIC")) idx_sic = i;
    }
    if (idx_firm < 0 || idx_sic < 0) die("Mapping CSV must have Firm_ID and SIC columns");

    long pos = ftell(fp);
    int N = 0;
    while (fgets(line, sizeof(line), fp)) N++;
    if (N <= 0) die("Mapping CSV has no data rows");
    fseek(fp, pos, SEEK_SET);

    mr->N = N;
    mr->firm_id = (char**)calloc(N, sizeof(char*));
    mr->sic = (char**)calloc(N, sizeof(char*));
    if (!mr->firm_id || !mr->sic) die("Out of memory for mapping arrays");

    int row = 0;
    while (row < N && fgets(line, sizeof(line), fp)) {
        int nr = split_csv_quoted_delim(line, tok, 2048, ',');
        if (nr <= idx_firm || nr <= idx_sic) die("Mapping row too short");

        char *fid = trim(tok[idx_firm]);
        char *sc  = trim(tok[idx_sic]);
        normalize_code_inplace(fid);
        normalize_code_inplace(sc);

        size_t lf = strlen(fid), ls = strlen(sc);
        mr->firm_id[row] = (char*)malloc(lf + 1);
        mr->sic[row] = (char*)malloc(ls + 1);
        if (!mr->firm_id[row] || !mr->sic[row]) die("Out of memory dup mapping fields");
        memcpy(mr->firm_id[row], fid, lf + 1);
        memcpy(mr->sic[row], sc, ls + 1);

        row++;
    }
    fclose(fp);
    fprintf(stdout, "[LOG] Mapping loaded: N=%d\n", mr->N);
}

static int sector_label_index(const IOTData *iot, const char *sic) {
    for (int j = 0; j < iot->S; ++j) {
        if (!strcmp(iot->labels[j], sic)) return j;
    }
    return -1;
}

static void build_firm_data(const FirmStrengthRaw *fr, const MappingRaw *mr,
                            const IOTData *iot, FirmData *fd,
                            int normalize_q) {
    if (fr->N != mr->N) die("Firm strength and mapping row counts differ (must be aligned)");
    int N = fr->N;
    fd->N = N;
    fd->q = (double*)calloc(N, sizeof(double));
    fd->sector = (int*)malloc((size_t)N * sizeof(int));
    if (!fd->q || !fd->sector) die("Out of memory building firm data");

    double qmax = 0.0;
    for (int i = 0; i < N; ++i) {
        if (strcmp(fr->firm_id[i], mr->firm_id[i]) != 0) {
            fprintf(stderr, "[FATAL] Firm_ID mismatch at row %d: '%s' vs '%s'\n",
                    i+1, fr->firm_id[i], mr->firm_id[i]);
            exit(EXIT_FAILURE);
        }
        fd->q[i] = fr->size[i];
        if (fd->q[i] > qmax) qmax = fd->q[i];
        fd->sector[i] = sector_label_index(iot, mr->sic[i]);
    }

    if (normalize_q) {
        if (qmax <= 0.0) die("Max firm size is non-positive; cannot normalize q");
        for (int i = 0; i < N; ++i) {
            fd->q[i] = fd->q[i] / qmax;
            if (fd->q[i] <= 0.0) fd->q[i] = 1e-16;
            if (fd->q[i] > 1.0) fd->q[i] = 1.0;
        }
        fprintf(stdout, "[LOG] q normalized by max(size)=%.6e\n", qmax);
    } else {
        fprintf(stdout, "[LOG] q NOT normalized (as-read)\n");
    }

    int mapped = 0;
    for (int i = 0; i < N; ++i) if (fd->sector[i] >= 0) mapped++;
    fprintf(stdout, "[LOG] Firms with mapped sector: %d / %d\n", mapped, N);

    if (mapped != N) die("Some firms have unmapped sectors. Every firm must belong to a sector.");
}

/* ---------- Read edges ---------- */

static void read_edges(const char *path, EdgeList *el, int flip_input, int N) {
    FILE *fp = fopen(path, "r");
    if (!fp) die_perror("Cannot open edges file", path);

    log_info("Reading edges...");
    int cap = 1 << 20;
    int *src = (int*)malloc((size_t)cap * sizeof(int));
    int *dst = (int*)malloc((size_t)cap * sizeof(int));
    if (!src || !dst) die("Out of memory reading edges");

    u64set_t seen;
    u64set_init(&seen, 2 * (size_t)cap);

    int m = 0;
    int u, v;
    while (fscanf(fp, "%d %d", &u, &v) == 2) {
        if (u <= 0 || v <= 0) die("Edges must be 1-based positive integers");
        int a = u - 1;
        int b = v - 1;

        if (a >= N || b >= N) {
            fprintf(stderr, "[FATAL] Edge %d->%d refers to node >= N=%d\n", u, v, N);
            exit(EXIT_FAILURE);
        }

        int s = flip_input ? b : a;
        int t = flip_input ? a : b;

        if (s == t) continue;

        uint64_t k = edge_key(s, t);
        if (u64set_has(&seen, k)) continue;
        u64set_add(&seen, k);

        if (m >= cap) {
            cap *= 2;
            src = (int*)realloc(src, (size_t)cap * sizeof(int));
            dst = (int*)realloc(dst, (size_t)cap * sizeof(int));
            if (!src || !dst) die("Out of memory growing edges");
        }
        src[m] = s;
        dst[m] = t;
        m++;
    }
    fclose(fp);
    u64set_free(&seen);

    el->n = N;
    el->m = m;
    el->src = src;
    el->dst = dst;
    fprintf(stdout, "[LOG] Edges loaded (deduped, no self-loops): n=%d, m=%d\n", el->n, el->m);
}

/* ---------- Build adjacency lists ---------- */

static void build_adj(const EdgeList *el, Adj *g) {
    g->n = el->n;
    g->m = el->m;
    g->head = (int*)malloc((size_t)g->n * sizeof(int));
    g->to   = (int*)malloc((size_t)g->m * sizeof(int));
    g->next = (int*)malloc((size_t)g->m * sizeof(int));
    if (!g->head || !g->to || !g->next) die("Out of memory building adjacency");

    for (int i = 0; i < g->n; ++i) g->head[i] = -1;
    for (int e = 0; e < g->m; ++e) {
        int u = el->src[e];
        int v = el->dst[e];
        g->to[e] = v;
        g->next[e] = g->head[u];
        g->head[u] = e;
    }
}

static void build_rev_adj(const EdgeList *el, Adj *rg) {
    rg->n = el->n;
    rg->m = el->m;
    rg->head = (int*)malloc((size_t)rg->n * sizeof(int));
    rg->to   = (int*)malloc((size_t)rg->m * sizeof(int));
    rg->next = (int*)malloc((size_t)rg->m * sizeof(int));
    if (!rg->head || !rg->to || !rg->next) die("Out of memory building reverse adjacency");

    for (int i = 0; i < rg->n; ++i) rg->head[i] = -1;
    for (int e = 0; e < rg->m; ++e) {
        int u = el->src[e], v = el->dst[e];
        rg->to[e] = u;
        rg->next[e] = rg->head[v];
        rg->head[v] = e;
    }
}

/* ---------- SCC (Kosaraju iterative) ---------- */

static int compute_scc_kosaraju(const Adj *g, const Adj *rg, int **out_comp) {
    int n = g->n;
    int *vis = (int*)calloc(n, sizeof(int));
    int *order = (int*)malloc((size_t)n * sizeof(int));
    int *stack = (int*)malloc((size_t)n * sizeof(int));
    int *itstk = (int*)malloc((size_t)n * sizeof(int));
    if (!vis || !order || !stack || !itstk) die("Out of memory in SCC");

    int ord_sz = 0;

    for (int start = 0; start < n; ++start) {
        if (vis[start]) continue;

        int top = 0;
        stack[top] = start;
        itstk[top] = g->head[start];
        vis[start] = 1;
        top++;

        while (top > 0) {
            int u = stack[top-1];
            int e = itstk[top-1];
            if (e != -1) {
                itstk[top-1] = g->next[e];
                int v = g->to[e];
                if (!vis[v]) {
                    stack[top] = v;
                    itstk[top] = g->head[v];
                    vis[v] = 1;
                    top++;
                }
            } else {
                top--;
                order[ord_sz++] = u;
            }
        }
    }

    int *comp = (int*)malloc((size_t)n * sizeof(int));
    if (!comp) die("Out of memory comp");
    for (int i = 0; i < n; ++i) comp[i] = -1;

    int K = 0;
    for (int idx = n - 1; idx >= 0; --idx) {
        int v0 = order[idx];
        if (comp[v0] != -1) continue;

        int top = 0;
        stack[top++] = v0;
        comp[v0] = K;

        while (top > 0) {
            int u = stack[--top];
            for (int e = rg->head[u]; e != -1; e = rg->next[e]) {
                int v = rg->to[e];
                if (comp[v] == -1) {
                    comp[v] = K;
                    stack[top++] = v;
                }
            }
        }
        K++;
    }

    free(vis); free(order); free(stack); free(itstk);
    *out_comp = comp;
    return K;
}

/* ---------- Components & condensation ---------- */

typedef struct {
    int *nodes;
    int count;
    int cap;
} Component;

typedef struct {
    int a;
    int b;
    int Kab;
    int Lab;
    int start_idx;
    int end_idx;
} PairInfo;

typedef struct {
    int src;
    int dst;
    int pair_id;
    int dst_sector;
    double delta;
} CandEdge;

static Component *build_components(int n, const int *comp, int K, int **out_comp_size) {
    Component *C = (Component*)calloc(K, sizeof(Component));
    int *csz = (int*)calloc(K, sizeof(int));
    if (!C || !csz) die("Out of memory building components");

    for (int v = 0; v < n; ++v) csz[comp[v]]++;

    for (int k = 0; k < K; ++k) {
        C[k].count = 0;
        C[k].cap = csz[k];
        C[k].nodes = (int*)malloc((size_t)csz[k] * sizeof(int));
        if (csz[k] > 0 && !C[k].nodes) die("Out of memory component nodes");
    }

    for (int v = 0; v < n; ++v) {
        int k = comp[v];
        C[k].nodes[C[k].count++] = v;
    }

    *out_comp_size = csz;
    return C;
}

static void build_comp_adj_from_keys(int K, const uint64_t *keys, int mkeys,
                                     int **out_head, int **out_to, int **out_nx) {
    int *head = (int*)malloc((size_t)K * sizeof(int));
    int *to   = (int*)malloc((size_t)mkeys * sizeof(int));
    int *nx   = (int*)malloc((size_t)mkeys * sizeof(int));
    if (!head || !to || !nx) die("Out of memory building comp adjacency");

    for (int i=0;i<K;i++) head[i] = -1;
    for (int e=0;e<mkeys;e++) {
        int a = (int)(keys[e] >> 32);
        int b = (int)(keys[e] & 0xffffffffu);
        to[e] = b;
        nx[e] = head[a];
        head[a] = e;
    }
    *out_head = head; *out_to = to; *out_nx = nx;
}

static void comp_reach_dfs(int K, const int *head, const int *to, const int *nx,
                           int start, uint8_t *vis) {
    int *st = (int*)malloc((size_t)K * sizeof(int));
    if (!st) die("Out of memory in reach dfs");
    int top = 0;
    st[top++] = start;
    vis[start] = 1;
    while (top) {
        int u = st[--top];
        for (int e = head[u]; e != -1; e = nx[e]) {
            int v = to[e];
            if (!vis[v]) { vis[v]=1; st[top++]=v; }
        }
    }
    free(st);
}

static int condensation_is_strongly_connected_with_P(
    int K, const uint64_t *keys, int mkeys,
    const PairInfo *P, int Pn
){
    int base_m = mkeys;
    int add_m  = Pn;
    int m = base_m + add_m;

    int *head = (int*)malloc((size_t)K * sizeof(int));
    int *to   = (int*)malloc((size_t)m * sizeof(int));
    int *nx   = (int*)malloc((size_t)m * sizeof(int));
    if (!head || !to || !nx) die("Out of memory in cond strong check");

    for (int i=0;i<K;i++) head[i] = -1;

    int eidx = 0;
    for (int e=0;e<mkeys;e++) {
        int a = (int)(keys[e] >> 32);
        int b = (int)(keys[e] & 0xffffffffu);
        to[eidx] = b;
        nx[eidx] = head[a];
        head[a]  = eidx;
        eidx++;
    }
    for (int p=0;p<Pn;p++) {
        int a = P[p].a, b = P[p].b;
        to[eidx] = b;
        nx[eidx] = head[a];
        head[a]  = eidx;
        eidx++;
    }

    int *rhead = (int*)malloc((size_t)K * sizeof(int));
    int *rto   = (int*)malloc((size_t)m * sizeof(int));
    int *rnx   = (int*)malloc((size_t)m * sizeof(int));
    if (!rhead || !rto || !rnx) die("Out of memory in cond strong check (rev)");

    for (int i=0;i<K;i++) rhead[i] = -1;
    int ridx = 0;
    for (int u=0;u<K;u++) {
        for (int e=head[u]; e!=-1; e=nx[e]) {
            int v = to[e];
            rto[ridx] = u;
            rnx[ridx] = rhead[v];
            rhead[v]  = ridx;
            ridx++;
        }
    }

    uint8_t *vis  = (uint8_t*)calloc((size_t)K, 1);
    uint8_t *rvis = (uint8_t*)calloc((size_t)K, 1);
    if (!vis || !rvis) die("Out of memory in cond strong check (vis)");

    int root = 0;
    comp_reach_dfs(K, head, to, nx, root, vis);
    comp_reach_dfs(K, rhead, rto, rnx, root, rvis);

    int ok = 1;
    for (int i=0;i<K;i++) {
        if (!vis[i] || !rvis[i]) { ok = 0; break; }
    }

    free(vis); free(rvis);
    free(head); free(to); free(nx);
    free(rhead); free(rto); free(rnx);
    return ok;
}

static void build_condensation_unique_edges(const EdgeList *el, const int *comp, int K,
                                            uint64_t **out_keys, int *out_mkeys,
                                            int **out_indeg, int **out_outdeg) {
    u64set_t set;
    u64set_init(&set, (size_t)el->m / 4 + 16);

    int *indeg = (int*)calloc(K, sizeof(int));
    int *outdeg = (int*)calloc(K, sizeof(int));
    if (!indeg || !outdeg) die("Out of memory indeg/outdeg");

    for (int e = 0; e < el->m; ++e) {
        int u = el->src[e], v = el->dst[e];
        int cu = comp[u], cv = comp[v];
        if (cu == cv) continue;
        uint64_t key = ((uint64_t)(uint32_t)cu << 32) | (uint64_t)(uint32_t)cv;
        if (!u64set_has(&set, key)) {
            u64set_add(&set, key);
        }
    }

    uint64_t *keys = (uint64_t*)malloc(set.n * sizeof(uint64_t));
    if (!keys) die("Out of memory keys");
    int mkeys = 0;
    for (size_t i = 0; i < set.cap; ++i) {
        if (set.used[i]) keys[mkeys++] = set.keys[i];
    }

    for (int i = 0; i < mkeys; ++i) {
        int cu = (int)(keys[i] >> 32);
        int cv = (int)(keys[i] & 0xffffffffu);
        outdeg[cu] += 1;
        indeg[cv] += 1;
    }

    u64set_free(&set);
    *out_keys = keys;
    *out_mkeys = mkeys;
    *out_indeg = indeg;
    *out_outdeg = outdeg;
}

static int *toposort_components(int K, const uint64_t *keys, int mkeys) {
    int *head = (int*)malloc((size_t)K * sizeof(int));
    int *to = (int*)malloc((size_t)mkeys * sizeof(int));
    int *nx = (int*)malloc((size_t)mkeys * sizeof(int));
    int *ind = (int*)calloc(K, sizeof(int));
    if (!head || !to || !nx || !ind) die("Out of memory building comp DAG");

    for (int i = 0; i < K; ++i) head[i] = -1;

    for (int e = 0; e < mkeys; ++e) {
        int a = (int)(keys[e] >> 32);
        int b = (int)(keys[e] & 0xffffffffu);
        to[e] = b;
        nx[e] = head[a];
        head[a] = e;
        ind[b]++;
    }

    int *q = (int*)malloc((size_t)K * sizeof(int));
    int *order = (int*)malloc((size_t)K * sizeof(int));
    if (!q || !order) die("Out of memory topo arrays");

    int qh = 0, qt = 0;
    for (int i = 0; i < K; ++i) if (ind[i] == 0) q[qt++] = i;

    int osz = 0;
    while (qh < qt) {
        int u = q[qh++];
        order[osz++] = u;
        for (int e = head[u]; e != -1; e = nx[e]) {
            int v = to[e];
            if (--ind[v] == 0) q[qt++] = v;
        }
    }

    if (osz != K) {
        die("Toposort failed; condensation not a DAG?");
    }

    free(head); free(to); free(nx); free(ind); free(q);
    return order;
}

static PairInfo *build_pairs_P(int K, const int *topo_order,
                               const int *is_source, const int *is_sink,
                               const uint64_t *ckeys, int mkeys,
                               int *out_Pn) {
    int ns = 0, nk = 0;
    for (int i = 0; i < K; ++i) {
        int c = topo_order[i];
        if (is_source[c]) ns++;
        if (is_sink[c])   nk++;
    }
    if (K == 1) { *out_Pn = 0; return NULL; }
    if (ns <= 0 || nk <= 0) die("Condensation must have >=1 source and >=1 sink when K>1");

    int *sources = (int*)malloc((size_t)ns * sizeof(int));
    int *sinks   = (int*)malloc((size_t)nk * sizeof(int));
    if (!sources || !sinks) die("Out of memory sources/sinks");

    int si = 0, ui = 0;
    for (int i = 0; i < K; ++i) {
        int c = topo_order[i];
        if (is_source[c]) sources[si++] = c;
        if (is_sink[c])   sinks[ui++]   = c;
    }

    int R = (ns > nk) ? ns : nk;

    for (int shift = 1; shift <= ns + 2; ++shift) {
        PairInfo *P = (PairInfo*)calloc((size_t)R, sizeof(PairInfo));
        if (!P) die("Out of memory PairInfo");

        int valid_shift = 1;
        for (int r = 0; r < R; ++r) {
            int a = sinks[r % nk];
            int b = sources[(r + shift) % ns];

            if (a == b) b = sources[(r + shift + 1) % ns];
            if (a == b) b = sources[(r + shift + 2) % ns];
            if (a == b) {
                free(P);
                P = NULL;
                valid_shift = 0;
                break;
            }

            P[r].a = a;
            P[r].b = b;
            P[r].Kab = 0;
            P[r].Lab = 0;
            P[r].start_idx = P[r].end_idx = 0;
        }

        if (!valid_shift || !P) continue;

        if (condensation_is_strongly_connected_with_P(K, ckeys, mkeys, P, R)) {
            free(sources); free(sinks);
            *out_Pn = R;
            return P;
        }

        free(P);
    }

    free(sources); free(sinks);
    die("Failed to build a valid P that makes condensation strongly connected.");
    return NULL;
}

/* ---------- Sector baselines B_ℓ and s_ℓ ---------- */

static void compute_sector_baselines(const FirmData *fd, const IOTData *iot,
                                     const EdgeList *el,
                                     double **out_B, double **out_s, double **out_base) {
    int S = iot->S;
    int N = fd->N;

    double *B = (double*)calloc(S, sizeof(double));
    double *s = (double*)calloc(S, sizeof(double));
    double *base = (double*)calloc(S, sizeof(double));
    if (!B || !s || !base) die("Out of memory baselines");

    for (int j = 0; j < N; ++j) {
        int lj = fd->sector[j];
        if (lj >= 0 && lj < S) s[lj] += fd->q[j];
    }

    for (int e = 0; e < el->m; ++e) {
        int i = el->src[e];
        int j = el->dst[e];
        if (i == j) continue;

        if (i < 0 || i >= N || j < 0 || j >= N) continue;
        int ki = fd->sector[i];
        int lj = fd->sector[j];
        if (ki >= 0 && ki < S && lj >= 0 && lj < S) {
            B[lj] += fd->q[i] * iot->Irow[ki * S + lj];
        }
    }

    for (int l = 0; l < S; ++l) base[l] = B[l] - s[l];

    *out_B = B;
    *out_s = s;
    *out_base = base;

    fprintf(stdout, "[LOG] Sector baselines computed (S=%d).\n", S);
}

/* ---------- Sampling candidates Ω_ab uniformly without replacement ---------- */

static void sample_absent_pairs_uniform(
    const Component *Ca, const Component *Cb,
    const u64set_t *Aset,
    rng64_t *rng,
    int Lab,
    int *out_src, int *out_dst,
    int *out_got
){
    u64set_t seen;
    u64set_init(&seen, (size_t)Lab + 16);

    int na = Ca->count, nb = Cb->count;
    int got = 0;

    uint64_t max_trials = 50ULL * (uint64_t)Lab;
    if (max_trials < 1000ULL) max_trials = 1000ULL;

    uint64_t trials = 0;
    while(got < Lab){
        if (++trials > max_trials) {
            die("Sampling Ω_ab: too many rejections. Likely Lab too close to absent count or orientation issue.");
        }

        int i = Ca->nodes[(int)uniform_u64_bounded(rng, (uint64_t)na)];
        int j = Cb->nodes[(int)uniform_u64_bounded(rng, (uint64_t)nb)];
        if(i==j) continue;

        uint64_t k = edge_key(i,j);
        if(u64set_has(Aset, k)) continue;
        if(u64set_has(&seen, k)) continue;

        u64set_add(&seen, k);
        out_src[got] = i;
        out_dst[got] = j;
        got++;
    }

    u64set_free(&seen);
    *out_got = got;
}

/* ---------- Build all candidates and pair-blocks ---------- */

static int ceil_int(double x) {
    int r = (int)ceil(x);
    return (r < 0) ? 0 : r;
}

static double f_eta(double theta, double eta, double n) {
    if (n <= 0.0) return 0.0;
    return theta * (1.0 - exp(-eta * n));
}

static double g_paper(double gamma_bar, double n0, double eta_pow, double n){
    if (n <= 0.0) return 0.0;
    double frac = n / (n0 + n);
    return gamma_bar * pow(frac, eta_pow);
}

static void build_candidates_all(const FirmData *fd, const IOTData *iot,
                                 const Component *C, const int *comp_size,
                                 PairInfo *P, int Pn,
                                 double theta, double eta, double gamma_bar, 
                                 double n0, double eta_g, const u64set_t *Aset,
                                 rng64_t *rng,
                                 const Adj *g, const int *comp,
                                 CandEdge **out_edges, int *out_M) {
    int S = iot->S;
    int N = fd->N;

    struct { int *src; int *dst; int count; } *sampled = malloc(Pn * sizeof(*sampled));
    if(!sampled && Pn > 0) die("Out of memory for sample structs");

    int M = 0;
    for (int p = 0; p < Pn; ++p) {
        sampled[p].src = NULL;
        sampled[p].dst = NULL;

        int a = P[p].a, b = P[p].b;
        int na = comp_size[a], nb = comp_size[b];
        int nab = (na < nb) ? na : nb;
        if (nab <= 0) die("Empty SCC unexpectedly");

        uint64_t exist_ab = 0;
        for (int i = 0; i < C[a].count; ++i) {
            int u = C[a].nodes[i];
            for (int e = g->head[u]; e != -1; e = g->next[e]) {
                int v = g->to[e];
                if (comp[v] == b) exist_ab++;
            }
        }
        uint64_t pop = (uint64_t)na * (uint64_t)nb;
        uint64_t absent_ab = pop - exist_ab;

        if (absent_ab == 0) {
            die("Pair (a,b) has zero absent edges; cannot sample from V(a)xV(b).");
        }

        int Kab = ceil_int(f_eta(theta, eta, (double)nab) * (double)nab);

        if (Kab < 1) Kab = 1;

        int Lab = ceil_int(g_paper(gamma_bar, n0, eta_g, (double)nab) * (double)nab);
        if (Lab < Kab) Lab = Kab;

        if ((uint64_t)Lab > absent_ab) Lab = (int)absent_ab;
        if (Kab > Lab) Kab = Lab;

        P[p].Kab = Kab;

        int got = 0;
        if (Lab > 0) {
            sampled[p].src = (int*)malloc((size_t)Lab * sizeof(int));
            sampled[p].dst = (int*)malloc((size_t)Lab * sizeof(int));
            if(!sampled[p].src || !sampled[p].dst) die("Out of memory sampling arrays");

            sample_absent_pairs_uniform(&C[a], &C[b], Aset, rng, Lab, sampled[p].src, sampled[p].dst, &got);

            if(got != Lab){
                die("Could not sample Lab absent edges; check orientation or density.");
            }

            P[p].Lab = got;
        } else {
            P[p].Lab = 0;
        }

        if (Kab < 1) {
            die("Kab dropped below 1 after capping by absent edges; need a different P pairing.");
        }

        P[p].start_idx = M;
        P[p].end_idx = M + got;
        sampled[p].count = got;
        M += got;
    }

    CandEdge *E = (CandEdge*)malloc((size_t)M * sizeof(CandEdge));
    if (!E && M > 0) die("Out of memory for candidate edges");

    for (int p = 0; p < Pn; ++p) {
        int base = P[p].start_idx;
        int count = sampled[p].count;

        for (int t = 0; t < count; ++t) {
            int src = sampled[p].src[t];
            int dst = sampled[p].dst[t];

#ifdef DEBUG
            if (u64set_has(Aset, edge_key(src, dst))) {
                die("Internal error: sampled candidate edge exists in original A");
            }
#endif

            int ks = (src >= 0 && src < N) ? fd->sector[src] : -1;
            int ld = (dst >= 0 && dst < N) ? fd->sector[dst] : -1;

            double delta = 0.0;
            if (ks >= 0 && ks < S && ld >= 0 && ld < S) {
                delta = fd->q[src] * iot->Irow[ks * S + ld];
            }

            E[base + t].src = src;
            E[base + t].dst = dst;
            E[base + t].pair_id = p;
            E[base + t].dst_sector = ld;
            E[base + t].delta = delta;
        }
        free(sampled[p].src);
        free(sampled[p].dst);
    }
    free(sampled);

    *out_edges = E;
    *out_M = M;
    fprintf(stdout, "[LOG] Candidates built: |P|=%d, total |Ω|=%d\n", Pn, M);
}

/* ---------- Solve MIQP with CPLEX ---------- */

static int solve_miqp(CandEdge *E, int M,
                      PairInfo *P, int Pn,
                      const double *base, const double *s, int S,
                      int **out_chosen) {
    int status = 0;
    CPXENVptr env = CPXopenCPLEX(&status);
    if (!env) die("CPXopenCPLEX failed");

    CPXLPptr lp = CPXcreateprob(env, &status, "markov_regular_3p3_faithful");
    if (!lp || status) die("CPXcreateprob failed");

    int nvars = M + S;
    double *obj = (double*)calloc((size_t)nvars, sizeof(double));
    double *lb  = (double*)malloc((size_t)nvars * sizeof(double));
    double *ub  = (double*)malloc((size_t)nvars * sizeof(double));
    char   *ctype = (char*)malloc((size_t)nvars * sizeof(char));
    if (!obj || !lb || !ub || !ctype) die("Out of memory for var arrays");

    for (int i = 0; i < M; ++i) {
        lb[i] = 0.0; ub[i] = 1.0; ctype[i] = 'B';
    }
    for (int l = 0; l < S; ++l) {
        lb[M + l] = -CPX_INFBOUND;
        ub[M + l] =  CPX_INFBOUND;
        ctype[M + l] = 'C';
    }

    status = CPXnewcols(env, lp, nvars, obj, lb, ub, ctype, NULL);
    if (status) die("CPXnewcols failed");

    for (int l = 0; l < S; ++l) {
        int idx = M + l;
        double sl = s[l];
        if (sl <= 0.0) die("Encountered sector with s_l <= 0; objective 1/s_l^2 undefined");
        double w = 1.0/(sl*sl);
        status = CPXchgqpcoef(env, lp, idx, idx, 2.0 * w);
        if (status) die("CPXchgqpcoef failed");
    }

    for (int p = 0; p < Pn; ++p) {
        int start = P[p].start_idx;
        int end   = P[p].end_idx;
        int nz = end - start;
        if (nz <= 0) {
             if (P[p].Kab == 0) continue;
             die("Pair has no candidates unexpectedly but Kab > 0");
        }

        int *ind = (int*)malloc((size_t)nz * sizeof(int));
        double *val = (double*)malloc((size_t)nz * sizeof(double));
        if (!ind || !val) die("Out of memory pair constraint");

        for (int k = 0; k < nz; ++k) {
            ind[k] = start + k;
            val[k] = 1.0;
        }
        double rhs = (double)P[p].Kab;
        char sense = 'E';
        int rmatbeg[1] = {0};

        status = CPXaddrows(env, lp, 0, 1, nz, &rhs, &sense, rmatbeg, ind, val, NULL, NULL);
        free(ind); free(val);
        if (status) die("CPXaddrows failed (pair constraint)");
    }

    int *cnt = (int*)calloc(S, sizeof(int));
    if (!cnt) die("Out of memory sector counts");
    for (int e = 0; e < M; ++e) {
        int l = E[e].dst_sector;
        if (l >= 0 && l < S) cnt[l]++;
    }
    int *start = (int*)malloc((size_t)(S + 1) * sizeof(int));
    if (!start) die("Out of memory sector start");
    start[0] = 0;
    for (int l = 0; l < S; ++l) start[l+1] = start[l] + cnt[l];
    int total = start[S];
    int *bucket = (int*)malloc((size_t)total * sizeof(int));
    int *fill = (int*)calloc(S, sizeof(int));
    if (!bucket || !fill) die("Out of memory bucket");

    for (int e = 0; e < M; ++e) {
        int l = E[e].dst_sector;
        if (l >= 0 && l < S) {
            int pos = start[l] + fill[l]++;
            bucket[pos] = e;
        }
    }

    for (int l = 0; l < S; ++l) {
        int nz_x = cnt[l];
        int nz = nz_x + 1;
        int *ind = (int*)malloc((size_t)nz * sizeof(int));
        double *val = (double*)malloc((size_t)nz * sizeof(double));
        if (!ind || !val) die("Out of memory y constraint");

        ind[0] = M + l;
        val[0] = 1.0;

        for (int t = 0; t < nz_x; ++t) {
            int e = bucket[start[l] + t];
            ind[1 + t] = e;
            val[1 + t] = -E[e].delta;
        }

        double rhs = base[l];
        char sense = 'E';
        int rmatbeg[1] = {0};

        status = CPXaddrows(env, lp, 0, 1, nz, &rhs, &sense, rmatbeg, ind, val, NULL, NULL);
        free(ind); free(val);
        if (status) die("CPXaddrows failed (y constraint)");
    }

    free(cnt); free(start); free(bucket); free(fill);

    CPXsetintparam(env, CPX_PARAM_THREADS, 0);
    CPXsetintparam(env, CPX_PARAM_MIPEMPHASIS, 1);

    log_info("Solving MIQP...");
    status = CPXmipopt(env, lp);
    if (status) die("CPXmipopt failed");

    int solstat = CPXgetstat(env, lp);
    fprintf(stdout, "[LOG] CPLEX status = %d\n", solstat);

    double objval = 0.0;
    if (!CPXgetobjval(env, lp, &objval)) {
        fprintf(stdout, "[LOG] CPLEX objective = %.12e\n", objval);
    }

    double *x = (double*)calloc((size_t)nvars, sizeof(double));
    if (!x) die("Out of memory solution");
    status = CPXgetx(env, lp, x, 0, nvars - 1);
    if (status) die("CPXgetx failed");

    int *chosen = (int*)calloc((size_t)M, sizeof(int));
    if (!chosen) die("Out of memory chosen");
    for (int e = 0; e < M; ++e) chosen[e] = (x[e] >= 0.5) ? 1 : 0;

    for (int p = 0; p < Pn; ++p) {
        int startp = P[p].start_idx;
        int endp = P[p].end_idx;
        int Kab = P[p].Kab;

        int picked = 0;
        for (int e = startp; e < endp; ++e) if (chosen[e]) picked++;

        if (picked != Kab) {
            fprintf(stderr, "[FATAL] Pair %d picked=%d but Kab=%d. CPLEX solution violates constraint.\n", 
                    p, picked, Kab);
            exit(EXIT_FAILURE);
        }
    }

    free(x);
    free(obj); free(lb); free(ub); free(ctype);

    CPXfreeprob(env, &lp);
    CPXcloseCPLEX(&env);

    *out_chosen = chosen;
    return 0;
}

/* ---------- Write output and SCC-check after augmentation ---------- */

static EdgeList build_augmented_edges(const EdgeList *orig, const CandEdge *E, int M, const int *chosen,
                                      int add_self_loops) {
    int n = orig->n;
    int add_edges = 0;
    if (M > 0) {
        if (!E || !chosen) die("Internal: M>0 but candidate arrays are NULL");
        for (int e = 0; e < M; ++e) if (chosen[e]) add_edges++;
    }

    int self = add_self_loops ? n : 0;
    int m2 = orig->m + add_edges + self;

    EdgeList out;
    out.n = n;
    out.m = m2;
    out.src = (int*)malloc((size_t)m2 * sizeof(int));
    out.dst = (int*)malloc((size_t)m2 * sizeof(int));
    if (!out.src || !out.dst) die("Out of memory augmented edges");

    int t = 0;
    for (int e = 0; e < orig->m; ++e) {
        out.src[t] = orig->src[e];
        out.dst[t] = orig->dst[e];
        t++;
    }
    for (int e = 0; e < M; ++e) {
        if (!chosen[e]) continue;
        out.src[t] = E[e].src;
        out.dst[t] = E[e].dst;
        t++;
    }
    if (add_self_loops) {
        for (int i = 0; i < n; ++i) {
            out.src[t] = i;
            out.dst[t] = i;
            t++;
        }
    }
    if (t != m2) die("Augmented edges internal count mismatch");
    return out;
}

static void write_edges_file(const char *path, const EdgeList *el, int flip_output) {
    FILE *fp = fopen(path, "w");
    if (!fp) die_perror("Cannot open output edges file", path);

    for (int e = 0; e < el->m; ++e) {
        int u = el->src[e] + 1;
        int v = el->dst[e] + 1;
        if (!flip_output)
            fprintf(fp, "%d %d\n", u, v);
        else
            fprintf(fp, "%d %d\n", v, u);
    }
    fclose(fp);
}

/* ---------- Main ---------- */

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s edges_unweighted.txt firm_strength.csv mapping.csv IOT.csv edges_irreducible.txt [options]\n\n"
        "Options:\n"
        "  --flip-input            interpret each line 'u v' as v->u\n"
        "  --flip-output           write each edge as 'dst src' instead of 'src dst'\n"
        "  --theta X               theta in f_eta(n)=theta*(1-exp(-eta*n)) (default 0.05)\n"
        "  --eta X                 eta   in f_eta (default 0.001)\n"
        "  --gamma X               gamma_bar in g_paper (default 0.25)\n"
        "  --n0 X                  n0 in g_paper (default 50.0)\n"
        "  --eta-g X               exponent in g_paper (default 1.0)\n"
        "  --seed U64              RNG seed (default 12345)\n"
        "  --normalize-q           normalize q by max(q)\n"
        "  --no-normalize-q        do NOT normalize q (default)\n",
        argv0
    );
}

int main(int argc, char **argv) {
    if (argc < 6) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *edges_in  = argv[1];
    const char *firm_csv  = argv[2];
    const char *map_csv   = argv[3];
    const char *iot_csv   = argv[4];
    const char *edges_out = argv[5];

    int flip_input = 0, flip_output = 0;
    double theta = 0.05;
    double eta   = 0.001;
    double gamma_bar = 0.25;
    double n0    = 50.0;
    double eta_g = 1.0;
    uint64_t seed = 12345;

    int normalize_q = 0;

    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--flip-input")) flip_input = 1;
        else if (!strcmp(argv[i], "--flip-output")) flip_output = 1;
        else if (!strcmp(argv[i], "--theta") && i+1 < argc) theta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eta")   && i+1 < argc) eta   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gamma") && i+1 < argc) gamma_bar = atof(argv[++i]);
        else if (!strcmp(argv[i], "--n0")    && i+1 < argc) n0    = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eta-g") && i+1 < argc) eta_g = atof(argv[++i]);
        else if (!strcmp(argv[i], "--zeta")) {
             fprintf(stderr, "[WARN] --zeta is deprecated, use --eta-g\n");
             if (i+1 < argc) ++i;
        }
        else if (!strcmp(argv[i], "--seed")  && i+1 < argc) seed  = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--normalize-q")) normalize_q = 1;
        else if (!strcmp(argv[i], "--no-normalize-q")) normalize_q = 0;
        else {
            fprintf(stderr, "[WARN] Unknown option: %s\n", argv[i]);
        }
    }

    if (!(theta > 0.0 && theta < 1.0)) die("theta must be in (0,1)");
    if (!(eta   > 0.0))                die("eta must be > 0");
    if (!(gamma_bar > 0.0))            die("gamma_bar must be > 0");
    if (!(n0 > 0.0))                   die("n0 must be > 0");
    if (!(eta_g > 0.0))                die("eta-g must be > 0");

    rng64_t rng = { seed };
    fprintf(stdout, "[LOG] RNG seed = %llu\n", (unsigned long long)seed);

    IOTData iot = {0};
    read_iot_csv(iot_csv, &iot);

    FirmStrengthRaw fr = {0};
    MappingRaw mr = {0};
    read_firm_strength_csv(firm_csv, &fr);
    read_mapping_csv(map_csv, &mr);

    FirmData fd = {0};
    build_firm_data(&fr, &mr, &iot, &fd, normalize_q);

    EdgeList el = {0};
    read_edges(edges_in, &el, flip_input, fd.N);

    u64set_t Aset;
    build_edge_set(&el, &Aset);

    Adj g = {0}, rg = {0};
    build_adj(&el, &g);
    build_rev_adj(&el, &rg);

    int *comp = NULL;
    int K = compute_scc_kosaraju(&g, &rg, &comp);
    fprintf(stdout, "[LOG] SCC count BEFORE augmentation: K=%d\n", K);
    if (K == 1) {
        log_info("Already strongly connected; only adding self-loops for aperiodicity.");

        EdgeList aug = build_augmented_edges(&el, NULL, 0, NULL, 1);
        write_edges_file(edges_out, &aug, flip_output);

        Adj g2 = {0}, rg2 = {0};
        build_adj(&aug, &g2);
        build_rev_adj(&aug, &rg2);
        int *comp2 = NULL;
        int K2 = compute_scc_kosaraju(&g2, &rg2, &comp2);
        fprintf(stdout, "[LOG] SCC count AFTER augmentation: K=%d\n", K2);

        free(comp2);
        free(g2.head); free(g2.to); free(g2.next);
        free(rg2.head); free(rg2.to); free(rg2.next);

        free(aug.src); free(aug.dst);

        free(comp);
        free(g.head); free(g.to); free(g.next);
        free(rg.head); free(rg.to); free(rg.next);
        free(el.src); free(el.dst);
        for (int j = 0; j < iot.S; ++j) free(iot.labels[j]);
        free(iot.labels); free(iot.Irow);
        for (int i = 0; i < fr.N; ++i) free(fr.firm_id[i]);
        free(fr.firm_id); free(fr.size);
        for (int i = 0; i < mr.N; ++i) { free(mr.firm_id[i]); free(mr.sic[i]); }
        free(mr.firm_id); free(mr.sic);
        free(fd.q); free(fd.sector);
        u64set_free(&Aset);

        return EXIT_SUCCESS;
    }

    int *comp_size = NULL;
    Component *C = build_components(g.n, comp, K, &comp_size);

    uint64_t *ckeys = NULL;
    int mkeys = 0;
    int *is_source = NULL, *is_sink = NULL;
    int *indeg_nonzero = NULL, *outdeg_nonzero = NULL;

    build_condensation_unique_edges(&el, comp, K, &ckeys, &mkeys, &indeg_nonzero, &outdeg_nonzero);

    is_source = (int*)calloc(K, sizeof(int));
    is_sink   = (int*)calloc(K, sizeof(int));
    if (!is_source || !is_sink) die("Out of memory source/sink flags");

    int ns = 0, nu = 0;
    for (int c = 0; c < K; ++c) {
        if (indeg_nonzero[c] == 0) { is_source[c] = 1; ns++; }
        if (outdeg_nonzero[c] == 0) { is_sink[c] = 1; nu++; }
    }
    fprintf(stdout, "[LOG] Condensation: sources=%d sinks=%d R=%d\n", ns, nu, (ns>nu)?ns:nu);

    int *topo = toposort_components(K, ckeys, mkeys);

    int Pn = 0;
    PairInfo *P = build_pairs_P(K, topo, is_source, is_sink, ckeys, mkeys, &Pn);
    fprintf(stdout, "[LOG] Built |P|=%d sink→source component pairs\n", Pn);

    double *B = NULL, *s = NULL, *base = NULL;
    compute_sector_baselines(&fd, &iot, &el, &B, &s, &base);

    CandEdge *Omega = NULL;
    int M = 0;
    build_candidates_all(&fd, &iot, C, comp_size, P, Pn,
                         theta, eta, gamma_bar, n0, eta_g, &Aset,
                         &rng, &g, comp,
                         &Omega, &M);

    int *chosen = NULL;
    solve_miqp(Omega, M, P, Pn, base, s, iot.S, &chosen);

    int picked_total = 0;
    for (int e = 0; e < M; ++e) if (chosen[e]) picked_total++;
    fprintf(stdout, "[LOG] Selected cross-SCC edges: %d\n", picked_total);

    EdgeList aug = build_augmented_edges(&el, Omega, M, chosen, 1);
    write_edges_file(edges_out, &aug, flip_output);
    fprintf(stdout, "[LOG] Written augmented edges to '%s'\n", edges_out);

    Adj g2 = {0}, rg2 = {0};
    build_adj(&aug, &g2);
    build_rev_adj(&aug, &rg2);
    int *comp2 = NULL;
    int K2 = compute_scc_kosaraju(&g2, &rg2, &comp2);
    fprintf(stdout, "[LOG] SCC count AFTER augmentation: K=%d\n", K2);
    if (K2 == 1) fprintf(stdout, "[LOG] SUCCESS: graph is irreducible (strongly connected).\n");
    else fprintf(stdout, "[WARN] Graph is still not strongly connected; check input orientation or constraints.\n");

    free(comp2);
    free(g2.head); free(g2.to); free(g2.next);
    free(rg2.head); free(rg2.to); free(rg2.next);

    free(aug.src); free(aug.dst);

    free(chosen);
    free(Omega);
    free(P);
    free(topo);
    free(is_source); free(is_sink);
    free(indeg_nonzero); free(outdeg_nonzero);
    free(ckeys);

    free(B); free(s); free(base);

    for (int k = 0; k < K; ++k) free(C[k].nodes);
    free(C);
    free(comp_size);

    free(comp);
    free(g.head); free(g.to); free(g.next);
    free(rg.head); free(rg.to); free(rg.next);

    free(el.src); free(el.dst);
    u64set_free(&Aset);

    for (int j = 0; j < iot.S; ++j) free(iot.labels[j]);
    free(iot.labels);
    free(iot.Irow);

    for (int i = 0; i < fr.N; ++i) free(fr.firm_id[i]);
    free(fr.firm_id);
    free(fr.size);

    for (int i = 0; i < mr.N; ++i) { free(mr.firm_id[i]); free(mr.sic[i]); }
    free(mr.firm_id);
    free(mr.sic);

    free(fd.q);
    free(fd.sector);

    log_info("Done.");
    return EXIT_SUCCESS;
}
