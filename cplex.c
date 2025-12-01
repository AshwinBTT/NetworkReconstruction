#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <ilcplex/cplex.h>

#define DIE(...) do{ fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1);}while(0)

/* --------------------------- I/O helpers --------------------------- */

typedef struct { int buyer; int *sup; int nSup; int rowOff; } Row;

static char* trim(char* s){
    while(isspace((unsigned char)*s)) ++s;
    if(!*s) return s;
    char* e=s+strlen(s)-1;
    while(e>s && isspace((unsigned char)*e)) --e;
    e[1]='\0';
    return s;
}

static int cmp_int(const void* a,const void* b){
    int x=*(const int*)a,y=*(const int*)b;
    return (x>y)-(x<y);
}

/* network: buyer, seller1, seller2, ... */
static Row* read_network(const char* path,int* out_nRows,int* out_maxID){
    FILE* f=fopen(path,"r"); if(!f) DIE("Cannot open %s",path);
    int cap=1024,n=0,maxID=0; Row* rows=(Row*)malloc(cap*sizeof(Row)); if(!rows) DIE("oom rows");
    char buf[1<<16];

    while(fgets(buf,sizeof buf,f)){
        char* l=trim(buf); if(!*l||*l=='#') continue;
        char* tok=strtok(l,","); if(!tok) continue;
        int b=atoi(tok); if(b<=0) continue;
        if(n==cap){ cap<<=1; rows=(Row*)realloc(rows,cap*sizeof(Row)); if(!rows) DIE("oom rows grow"); }
        rows[n].buyer=b; rows[n].nSup=0; rows[n].rowOff=-1;
        int sc=8; rows[n].sup=(int*)malloc(sc*sizeof(int)); if(!rows[n].sup) DIE("oom sup");
        while((tok=strtok(NULL,","))){
            int s=atoi(trim(tok)); if(s<=0) continue;
            if(rows[n].nSup==sc){
                sc<<=1;
                rows[n].sup=(int*)realloc(rows[n].sup,sc*sizeof(int));
                if(!rows[n].sup) DIE("oom sup grow");
            }
            rows[n].sup[ rows[n].nSup++ ] = s;
            if(s>maxID) maxID=s;
        }
        if(rows[n].nSup>0){
            qsort(rows[n].sup,rows[n].nSup,sizeof(int),cmp_int);
            int w=0;
            for(int i=1;i<rows[n].nSup;i++)
                if(rows[n].sup[i]!=rows[n].sup[w]) rows[n].sup[++w]=rows[n].sup[i];
            rows[n].nSup=w+1;
        }
        if(b>maxID) maxID=b;
        n++;
    }
    fclose(f);
    *out_nRows=n; *out_maxID=maxID; return rows;
}

/* money.txt: firmID, m_i */
static double* read_money(const char* path,int maxID,int* out_max_seen){
    FILE* f=fopen(path,"r"); if(!f) DIE("Cannot open %s",path);
    double* m=(double*)calloc((size_t)(maxID+1),sizeof(double)); if(!m) DIE("oom m");
    int max_seen=0; char buf[1<<16];

    while(fgets(buf,sizeof buf,f)){
        char* l=trim(buf); if(!*l||*l=='#') continue;
        int id; double val;
        if(sscanf(l,"%d,%lf",&id,&val)==2){
            if(id>=1 && id<=maxID) m[id] = (val>0.0? val:0.0);
            if(id>max_seen) max_seen=id;
        }
    }

    fclose(f);
    *out_max_seen=max_seen; return m;
}

/* firmSector.txt: firmID, sectorID  (sector codes can be non-contiguous ints) */
static int* read_sector_map(const char* path,int maxID,int* out_S, int** out_sector_ids){
    FILE* f=fopen(path,"r"); if(!f) DIE("Cannot open %s",path);
    int* f2s=(int*)malloc((size_t)(maxID+1)*sizeof(int)); if(!f2s) DIE("oom f2s");
    for(int i=0;i<=maxID;i++) f2s[i]=-1;

    int cap=1024,n=0; int* allsec=(int*)malloc(cap*sizeof(int)); if(!allsec) DIE("oom allsec");
    char buf[1<<16];

    while(fgets(buf,sizeof buf,f)){
        char* l=trim(buf); if(!*l||*l=='#') continue;
        int id, sec;
        if(sscanf(l,"%d,%d",&id,&sec)==2){
            if(id < 1 || id > maxID) continue;
            f2s[id]=sec;
            if(n==cap){
                cap<<=1;
                allsec=(int*)realloc(allsec,cap*sizeof(int));
                if(!allsec) DIE("oom allsec grow");
            }
            allsec[n++]=sec;
        }
    }
    fclose(f);
    if(n==0) DIE("firmSector file empty?");

    qsort(allsec,n,sizeof(int),cmp_int);
    int S=0;
    for(int i=0;i<n;i++) if(i==0 || allsec[i]!=allsec[i-1]) allsec[S++]=allsec[i];

    int* sector_ids=(int*)malloc(S*sizeof(int)); if(!sector_ids) DIE("oom sector_ids");
    for(int i=0;i<S;i++) sector_ids[i]=allsec[i];
    free(allsec);

    *out_S=S; *out_sector_ids=sector_ids; return f2s;
}

/* optional per-firm deltas: id, delta_j ; otherwise default */
static double* read_firm_deltas(const char* path,int maxID, double default_delta){
    FILE* f=fopen(path,"r");
    double* d=(double*)malloc((size_t)(maxID+1)*sizeof(double)); if(!d) DIE("oom deltas");
    for(int i=0;i<=maxID;i++) d[i]=default_delta;
    if(!f) return d;

    char buf[1<<16];
    while(fgets(buf,sizeof buf,f)){
        char* l=trim(buf); if(!*l||*l=='#') continue;
        int id; double dv;
        if(sscanf(l,"%d,%lf",&id,&dv)==2){
            if(id>=1 && id<=maxID && dv>=0.0) d[id]=dv;
        }
    }
    fclose(f);
    return d;
}

/* sector code -> position 0..S-1 (sector_ids sorted) */
static int sector_pos(int sec, const int* ids, int S){
    int lo=0, hi=S-1;
    while(lo<=hi){
        int mid=(lo+hi)>>1, v=ids[mid];
        if(v==sec) return mid;
        if(v<sec) lo=mid+1; else hi=mid-1;
    }
    return -1;
}

/* CPLEX error checking helper */
static void cpx_chk(CPXENVptr env, int status, const char* what){
    if(status){
        char msg[1024] = {0};
        CPXgeterrorstring(env, status, msg);
        fprintf(stderr, "[CPLEX] %s failed (%d): %s\n", what, status, msg);
        exit(1);
    }
}
#define CPXCHK(call) do{ int _s=(call); cpx_chk(env,_s,#call); }while(0)

/* Add a single ranged row: lhs <= sum_k val[k]*x[ind[k]] <= rhs */
static void add_ranged_row(CPXENVptr env, CPXLPptr lp,
                           int nnz, const int* ind, const double* val,
                           double lhs, double rhs)
{
    if (rhs < lhs) DIE("add_ranged_row: rhs < lhs (%.17g < %.17g)", rhs, lhs);

    double rowrhs = lhs;
    char sense = 'R';
    int rbeg = 0;

    int st = CPXaddrows(env, lp,
                        0, 1, nnz,
                        &rowrhs, &sense,
                        &rbeg,
                        (int*)ind, (double*)val,
                        NULL, NULL);
    if (st) DIE("CPXaddrows(ranged) failed (%d)", st);

    int rowind = CPXgetnumrows(env, lp) - 1;
    double rng = rhs - lhs;
    st = CPXchgrngval(env, lp, 1, &rowind, &rng);
    if (st) DIE("CPXchgrngval failed (%d)", st);
}

/* ------------------------------ Main ------------------------------ */

int main(int argc,char** argv){
    if(argc < 5){
        fprintf(stderr,
                "Usage: %s edges_irreducible.txt money.txt firmSector.txt weights_out.csv "
                "[--firm_rel r | --firm_rel_file firm_deltas.csv] "
                "--sector_rel eps --eta1 e1 --eta2 e2 [--eps0 x]\n",
                argv[0]);
        return 1;
    }

    const char* FNET  = argv[1];
    const char* FMONEY= argv[2];
    const char* FSECT = argv[3];
    const char* FOUT  = argv[4];

    double firm_rel_default = 0.10;
    const char* firm_delta_file = NULL;
    double sector_rel_eps = 0.10;
    double eta1 = 0.10;
    double eta2 = 0.10;
    double eps0 = 0.0;

    for(int a=5;a<argc;a++){
        if(strcmp(argv[a],"--firm_rel")==0 && a+1<argc){
            firm_rel_default = strtod(argv[++a],NULL);
        } else if(strcmp(argv[a],"--firm_rel_file")==0 && a+1<argc){
            firm_delta_file = argv[++a];
        } else if(strcmp(argv[a],"--sector_rel")==0 && a+1<argc){
            sector_rel_eps = strtod(argv[++a],NULL);
        } else if(strcmp(argv[a],"--eta1")==0 && a+1<argc){
            eta1 = strtod(argv[++a],NULL);
        } else if(strcmp(argv[a],"--eta2")==0 && a+1<argc){
            eta2 = strtod(argv[++a],NULL);
        } else if(strcmp(argv[a],"--eps0")==0 && a+1<argc){
            eps0 = strtod(argv[++a],NULL);
        } else {
            fprintf(stderr,"Unknown/ill-formed arg near %s\n", argv[a]);
            return 1;
        }
    }

    if(firm_rel_default < 0.0 || sector_rel_eps < 0.0 || eta1 < 0.0 || eta2 < 0.0 || eps0 < 0.0)
        DIE("Negative tolerances not allowed.");

    int nRows=0, maxID_net=0;
    Row* rows = read_network(FNET, &nRows, &maxID_net);
    if(nRows==0) DIE("Empty network.");
    for(int r=0;r<nRows;r++) if(rows[r].nSup<=0) DIE("Buyer %d has no suppliers.", rows[r].buyer);

    {
        int* seenBuyer = (int*)calloc((size_t)(maxID_net+1), sizeof(int));
        if(!seenBuyer) DIE("oom seenBuyer");
        for(int r=0;r<nRows;r++){
            int b = rows[r].buyer;
            if(b>=1 && b<=maxID_net){
                if(seenBuyer[b]) DIE("Duplicate buyer row in network for firm %d", b);
                seenBuyer[b] = 1;
            }
        }
        free(seenBuyer);
    }

    int* inNet = (int*)calloc((size_t)(maxID_net+1), sizeof(int));
    if(!inNet) DIE("oom inNet");
    for(int r=0;r<nRows;r++){
        int b = rows[r].buyer;
        if(b>=1 && b<=maxID_net) inNet[b]=1;
        for(int t=0;t<rows[r].nSup;t++){
            int s = rows[r].sup[t];
            if(s>=1 && s<=maxID_net) inNet[s]=1;
        }
    }

    if (eps0 > 0.0) {
        for(int r=0;r<nRows;r++){
            int d = rows[r].nSup;
            if ((double)d * eps0 > 1.0 + 1e-12) {
                DIE("Infeasible: buyer %d has outdegree=%d but d*eps0=%.6g>1. Reduce eps0 or increase outdegree.",
                    rows[r].buyer, d, (double)d*eps0);
            }
        }
    }

    int max_money_id=0;
    double* m = read_money(FMONEY, maxID_net, &max_money_id);
    int maxID = maxID_net;

    if (max_money_id > maxID_net) {
        DIE("money.txt contains firm id %d > maxID inferred from network (%d). Fix inputs or expand network ids.",
            max_money_id, maxID_net);
    }

    int S=0; int* sector_ids=NULL;
    int* firm2sec = read_sector_map(FSECT, maxID, &S, &sector_ids);
    if(S<=0) DIE("No sectors detected.");

    {
        int* buyer2row=(int*)malloc((size_t)(maxID+1)*sizeof(int));
        if(!buyer2row) DIE("oom buyer2row");
        for(int i=0;i<=maxID;i++) buyer2row[i]=-1;
        for(int r=0;r<nRows;r++){
            int b=rows[r].buyer;
            if(b>=1 && b<=maxID) buyer2row[b]=r;
        }

        for(int i=1;i<=maxID;i++){
            if(buyer2row[i]>=0 && firm2sec[i]==-1) DIE("Firm %d in network but missing sector code.", i);
            if(inNet[i] && buyer2row[i]<0) DIE("Firm %d appears in network (as buyer or seller) but has no buyer row.", i);
        }
        free(buyer2row);
    }

    double* delta = read_firm_deltas(firm_delta_file, maxID, firm_rel_default);

    double mmax=0.0;
    for(int i=1;i<=maxID;i++) if(m[i]>mmax) mmax=m[i];
    if(mmax>0.0){
        double c = 1.0/mmax;
        for(int i=1;i<=maxID;i++) m[i]*=c;
    }

    {
        for(int v=1; v<=maxID; v++){
            if(inNet[v] && m[v] <= 0.0){
                DIE("Firm %d appears in network but has m<=0. Remove it earlier or assign positive size.", v);
            }
        }
    }

    long long tot=0;
    for(int r=0;r<nRows;r++) tot += rows[r].nSup;
    if(tot > 2147483647LL) DIE("Too many edges.");
    const int NE = (int)tot;

    int *eBuyer=(int*)malloc(NE*sizeof(int));
    int *eSeller=(int*)malloc(NE*sizeof(int));
    if(!eBuyer||!eSeller) DIE("oom edges");

    int* colcnt = (int*)calloc((size_t)(maxID+1), sizeof(int));
    int** colind = (int**)malloc((size_t)(maxID+1)*sizeof(int*));
    int* colpos = (int*)calloc((size_t)(maxID+1), sizeof(int));
    if(!colcnt||!colind||!colpos) DIE("oom col arrays");

    for(int r=0;r<nRows;r++){
        for(int t=0;t<rows[r].nSup;t++){
            int j = rows[r].sup[t];
            if(j>=1 && j<=maxID) colcnt[j]++;
        }
    }
    for(int j=1;j<=maxID;j++){
        if(colcnt[j]>0){
            colind[j]=(int*)malloc((size_t)colcnt[j]*sizeof(int));
            if(!colind[j]) DIE("oom colind[j]");
        } else colind[j]=NULL;
    }

    int k=0;
    for(int r=0;r<nRows;r++){
        rows[r].rowOff = k;
        int i = rows[r].buyer;
        for(int t=0;t<rows[r].nSup;t++,k++){
            int j = rows[r].sup[t];
            eBuyer[k]=i; eSeller[k]=j;
            int p = colpos[j]++;
            colind[j][p] = k;
        }
    }

    int* sellerSecPos = (int*)malloc((size_t)(maxID+1)*sizeof(int));
    if(!sellerSecPos) DIE("oom sellerSecPos");
    for(int j=0;j<=maxID;j++) sellerSecPos[j] = -1;
    for(int j=1;j<=maxID;j++){
        if(!inNet[j]) continue;
        if(firm2sec[j]!=-1){
            sellerSecPos[j] = sector_pos(firm2sec[j], sector_ids, S);
        }
    }

    int* secEdgeCnt = (int*)calloc(S, sizeof(int));
    if(!secEdgeCnt) DIE("oom secEdgeCnt");
    for(int e=0;e<NE;e++){
        int j = eSeller[e];
        if(j<1 || j>maxID) continue;
        if(!inNet[j]) continue;
        int l = sellerSecPos[j];
        if(l>=0) secEdgeCnt[l]++;
    }

    int** secEdges = (int**)malloc(S*sizeof(int*));
    int* secFill  = (int*)calloc(S, sizeof(int));
    if(!secEdges||!secFill) DIE("oom secEdges/secFill");
    for(int l=0;l<S;l++){
        secEdges[l] = (secEdgeCnt[l]>0) ? (int*)malloc((size_t)secEdgeCnt[l]*sizeof(int)) : NULL;
        if(secEdgeCnt[l]>0 && !secEdges[l]) DIE("oom secEdges[l]");
    }
    for(int e=0;e<NE;e++){
        int j = eSeller[e];
        if(j<1 || j>maxID) continue;
        if(!inNet[j]) continue;
        int l = sellerSecPos[j];
        if(l>=0) secEdges[l][ secFill[l]++ ] = e;
    }

    int** firmsInSec = (int**)malloc(S*sizeof(int*));
    int* secCnt     = (int*)calloc(S, sizeof(int));
    if(!firmsInSec||!secCnt) DIE("oom sec arrays");

    for(int f=1; f<=maxID; ++f)
        if(inNet[f] && firm2sec[f]!=-1){
            int pos = sector_pos(firm2sec[f], sector_ids, S);
            if(pos>=0) secCnt[pos]++;
        }

    for(int sct=0;sct<S;sct++){
        firmsInSec[sct] = (secCnt[sct]>0? (int*)malloc(secCnt[sct]*sizeof(int)) : NULL);
        if(secCnt[sct]>0 && !firmsInSec[sct]) DIE("oom firmsInSec[sct]");
        secCnt[sct] = 0;
    }
    for(int f=1; f<=maxID; ++f)
        if(inNet[f] && firm2sec[f]!=-1){
            int pos = sector_pos(firm2sec[f], sector_ids, S);
            if(pos>=0) firmsInSec[pos][ secCnt[pos]++ ] = f;
        }

    double* s_tot = (double*)calloc(S, sizeof(double));
    if(!s_tot) DIE("oom s_tot");
    for(int l=0;l<S;l++){
        double sum=0.0;
        for(int u=0; u<secCnt[l]; ++u){
            int j = firmsInSec[l][u];
            sum += m[j];
        }
        s_tot[l]=sum;
    }

    for(int l=0; l<S; l++){
        if(s_tot[l] <= 0.0 && secCnt[l] > 0){
            DIE("Sector code %d has firms but total size s_l<=0 after scaling. Assign positive m for firms in this sector.",
                sector_ids[l]);
        }
    }

    int* seen = (int*)calloc((size_t)(maxID+1), sizeof(int)); if(!seen) DIE("oom seen");
    for(int r=0; r<nRows; ++r){
        int i = rows[r].buyer; if(i>=1 && i<=maxID) seen[i] = 1;
        for(int t=0;t<rows[r].nSup;t++){
            int j = rows[r].sup[t]; if(j>=1 && j<=maxID) seen[j] = 1;
        }
    }
    int N_F = 0; for(int v=1; v<=maxID; ++v) if(seen[v]) N_F++;
    free(seen);

    int st=0; CPXENVptr env = CPXopenCPLEX(&st);
    if(!env) DIE("CPXopenCPLEX failed (%d)", st);

    CPXCHK(CPXsetintparam(env, CPX_PARAM_SCRIND,         CPX_ON));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_THREADS,        0));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_NUMERICALEMPHASIS, 1));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_QPMETHOD,       0));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_BARORDER,       0));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_PREIND,         CPX_ON));
    CPXCHK(CPXsetintparam(env, CPX_PARAM_SCAIND,         1));
    CPXCHK(CPXsetdblparam(env, CPX_PARAM_EPOPT,          1e-6));
    CPXCHK(CPXsetdblparam(env, CPX_PARAM_EPRHS,          1e-6));
    CPXCHK(CPXsetdblparam(env, CPX_PARAM_BAREPCOMP,      1e-6));
    CPXCHK(CPXsetintparam(env, CPXPARAM_Emphasis_Memory, 1));
    CPXCHK(CPXsetdblparam(env, CPXPARAM_WorkMem, 64000.0));

    CPXLPptr lp = CPXcreateprob(env, &st, "min_energy_qcp");
    if(!lp) DIE("CPXcreateprob failed (%d)", st);
    CPXchgobjsen(env, lp, CPX_MIN);

    const int NCOLS = NE;

    double* obj = (double*)calloc(NCOLS, sizeof(double));
    double* lb  = (double*)malloc(NCOLS * sizeof(double));
    double* ub  = (double*)malloc(NCOLS * sizeof(double));
    if(!obj||!lb||!ub) DIE("oom col arrays");

    for(int e=0; e<NE; ++e){ lb[e]=eps0; ub[e]=1.0; obj[e]=0.0; }

    st = CPXnewcols(env, lp, NCOLS, obj, lb, ub, NULL, NULL);
    if(st) DIE("CPXnewcols failed (%d)", st);
    free(obj); free(lb); free(ub);

    {
        int rows_to_add = nRows;
        int nnz = NE;

        double *rhs = (double*)malloc((size_t)rows_to_add * sizeof(double));
        char   *sense = (char*)malloc((size_t)rows_to_add * sizeof(char));
        int    *rmatbeg = (int*)malloc((size_t)rows_to_add * sizeof(int));
        int    *rmatind = (int*)malloc((size_t)nnz * sizeof(int));
        double *rmatval = (double*)malloc((size_t)nnz * sizeof(double));
        if(!rhs||!sense||!rmatbeg||!rmatind||!rmatval) DIE("oom batch RS");

        int pos = 0;
        for(int r=0;r<nRows;r++){
            rhs[r] = 1.0;
            sense[r] = 'E';
            rmatbeg[r] = pos;
            int d = rows[r].nSup;
            for(int t=0;t<d;t++){
                rmatind[pos] = rows[r].rowOff + t;
                rmatval[pos] = 1.0;
                pos++;
            }
        }
        if(pos != nnz) DIE("batch RS: pos(%d) != NE(%d)", pos, nnz);

        st = CPXaddrows(env, lp, 0, rows_to_add, nnz, rhs, sense, rmatbeg, rmatind, rmatval, NULL, NULL);
        if(st) DIE("CPXaddrows(batch RS) failed (%d)", st);

        free(rhs); free(sense); free(rmatbeg); free(rmatind); free(rmatval);
    }

    {
        int selfCnt=0;
        for(int e=0;e<NE;e++) if(eBuyer[e]==eSeller[e]) selfCnt++;
        if(selfCnt>0){
            int* ind=(int*)malloc(selfCnt*sizeof(int));
            double* val=(double*)malloc(selfCnt*sizeof(double));
            if(!ind||!val) DIE("oom self lin");
            int q=0;
            for(int e=0;e<NE;e++)
                if(eBuyer[e]==eSeller[e]){ ind[q]=e; val[q]=1.0; q++; }
            double rhs = (double)N_F * eta1; char sense='L'; int rbeg=0;
            st=CPXaddrows(env, lp, 0, 1, selfCnt, &rhs, &sense, &rbeg, ind, val, NULL, NULL);
            if(st) DIE("add self mean failed (%d)", st);
            free(ind); free(val);
        }
    }

    {
        int *firmRowIndex = (int*)malloc((size_t)(maxID+1) * sizeof(int));
        if(!firmRowIndex) DIE("oom firmRowIndex");
        for(int j=0;j<=maxID;j++) firmRowIndex[j] = -1;

        int nFirmRows = 0;
        long long firmNNZ = 0;

        for(int j=1;j<=maxID;j++){
            if(!inNet[j]) continue;

            if(colcnt[j]==0){
                if(m[j]==0.0) continue;
                if(delta[j] < 1.0 - 1e-15){
                    DIE("Infeasible: firm %d has no incoming edges but m_j>0 and delta_j=%.6g < 1", j, delta[j]);
                }
                continue;
            }

            firmRowIndex[j] = nFirmRows++;
            firmNNZ += colcnt[j];
        }

        if(nFirmRows > 0){
            double *rhs = (double*)malloc((size_t)nFirmRows * sizeof(double));
            char   *sense = (char*)malloc((size_t)nFirmRows * sizeof(char));
            int    *rmatbeg = (int*)malloc((size_t)nFirmRows * sizeof(int));
            int    *rmatind = (int*)malloc((size_t)firmNNZ * sizeof(int));
            double *rmatval = (double*)malloc((size_t)firmNNZ * sizeof(double));
            if(!rhs||!sense||!rmatbeg||!rmatind||!rmatval) DIE("oom batch firm rows");

            int pos = 0;
            for(int j=1;j<=maxID;j++){
                int ridx = firmRowIndex[j];
                if(ridx < 0) continue;

                rmatbeg[ridx] = pos;

                for(int p=0;p<colcnt[j];p++){
                    int e = colind[j][p];
                    rmatind[pos] = e;
                    rmatval[pos] = m[eBuyer[e]];
                    pos++;
                }

                if(m[j] == 0.0){
                    rhs[ridx] = 0.0;
                    sense[ridx] = 'R';
                } else {
                    double dj = delta[j];
                    double rhsL = (1.0 - dj) * m[j];
                    rhs[ridx] = rhsL;
                    sense[ridx] = 'R';
                }
            }

            if(pos != firmNNZ) DIE("batch firm: pos(%d) != firmNNZ(%lld)", pos, firmNNZ);

            st = CPXaddrows(env, lp, 0, nFirmRows, (int)firmNNZ, rhs, sense, rmatbeg, rmatind, rmatval, NULL, NULL);
            if(st) DIE("CPXaddrows(batch firm) failed (%d)", st);

            int baseRow = CPXgetnumrows(env, lp) - nFirmRows;
            for(int j=1;j<=maxID;j++){
                int ridx = firmRowIndex[j];
                if(ridx < 0) continue;

                double lhs, up;
                if(m[j] == 0.0){
                    lhs = 0.0; up = 0.0;
                } else {
                    double dj = delta[j];
                    lhs = (1.0 - dj) * m[j];
                    up  = (1.0 + dj) * m[j];
                }
                int rowind = baseRow + ridx;
                double rng = up - lhs;
                st = CPXchgrngval(env, lp, 1, &rowind, &rng);
                if(st) DIE("CPXchgrngval(batch firm) failed (%d)", st);
            }

            free(rhs); free(sense); free(rmatbeg); free(rmatind); free(rmatval);
        }

        free(firmRowIndex);
    }

    for(int l=0;l<S;l++){
        int nnz = secEdgeCnt[l];
        if(nnz==0){
            if(s_tot[l]==0.0) continue;

            if(sector_rel_eps < 1.0 - 1e-15){
                DIE("Infeasible: sector %d has no edges but s_l>0 and eps=%.6g < 1", sector_ids[l], sector_rel_eps);
            }
            continue;
        }

        if(s_tot[l] == 0.0){
            if(nnz==0) continue;
            int* ind = (int*)malloc(nnz*sizeof(int));
            double* val = (double*)malloc(nnz*sizeof(double));
            if(!ind||!val) DIE("oom sect Shat=0");
            for(int p=0;p<nnz;p++){
                int e = secEdges[l][p];
                int i = eBuyer[e];
                ind[p] = e;
                val[p] = m[i];
            }
            double rhs=0.0; char sense='E'; int rbeg=0;
            st = CPXaddrows(env, lp, 0, 1, nnz, &rhs, &sense, &rbeg, ind, val, NULL, NULL);
            if(st) DIE("add sector Shat=0 failed (%d)", st);
            free(ind); free(val);
            continue;
        }

        double rhsU = (1.0 + sector_rel_eps) * s_tot[l];
        double rhsL = (1.0 - sector_rel_eps) * s_tot[l];

        int* ind = (int*)malloc(nnz*sizeof(int));
        double* val = (double*)malloc(nnz*sizeof(double));
        if(!ind||!val) DIE("oom sector band");
        for(int p=0;p<nnz;p++){
            int e = secEdges[l][p];
            int i = eBuyer[e];
            ind[p] = e;
            val[p] = m[i];
        }

        add_ranged_row(env, lp, nnz, ind, val, rhsL, rhsU);
        free(ind); free(val);
    }

    {
        int* qmatbeg = (int*)malloc(NE*sizeof(int));
        int* qmatcnt = (int*)malloc(NE*sizeof(int));
        int* qmatind = (int*)malloc(NE*sizeof(int));
        double* qmatval = (double*)malloc(NE*sizeof(double));
        if(!qmatbeg||!qmatcnt||!qmatind||!qmatval) DIE("oom Q objective arrays");

        for(int e=0;e<NE;e++){
            qmatbeg[e] = e;
            qmatcnt[e] = 1;
            qmatind[e] = e;
            qmatval[e] = 2.0;
        }
        st = CPXcopyquad(env, lp, qmatbeg, qmatcnt, qmatind, qmatval);
        if(st) DIE("CPXcopyquad failed (%d)", st);

        free(qmatbeg); free(qmatcnt); free(qmatind); free(qmatval);
    }

    st = CPXqpopt(env, lp);
    if(st) DIE("CPXqpopt failed (%d)", st);

    double* x_qp = (double*)malloc((size_t)NE*sizeof(double));
    if(!x_qp) DIE("oom x_qp");
    st = CPXgetx(env, lp, x_qp, 0, NE-1);
    if(st) DIE("CPXgetx(QP) failed (%d)", st);

    double selfSq = 0.0;
    for(int e=0;e<NE;e++){
        if(eBuyer[e]==eSeller[e]){
            double a = x_qp[e];
            selfSq += a*a;
        }
    }

    double capSelfSq = (double)N_F * eta2;
    const double SQTOL = 1e-12;

    if(selfSq > capSelfSq + SQTOL){
        int selfCnt=0;
        for(int e=0;e<NE;e++) if(eBuyer[e]==eSeller[e]) selfCnt++;

        if(selfCnt>0){
            int* quadrow = (int*)malloc(selfCnt*sizeof(int));
            int* quadcol = (int*)malloc(selfCnt*sizeof(int));
            double* quadval = (double*)malloc(selfCnt*sizeof(double));
            if(!quadrow||!quadcol||!quadval) DIE("oom self qcp triplets");

            int w=0;
            for(int e=0;e<NE;e++) if(eBuyer[e]==eSeller[e]){
                quadrow[w] = e;
                quadcol[w] = e;
                quadval[w] = 1.0;
                w++;
            }

            st = CPXaddqconstr(env, lp,
                               0,
                               selfCnt,
                               capSelfSq,
                               'L',
                               NULL, NULL,
                               quadrow, quadcol, quadval,
                               "self_sq");
            if(st) DIE("CPXaddqconstr(self_sq) failed (%d)", st);

            free(quadrow); free(quadcol); free(quadval);
        }

        st = CPXbaropt(env, lp);
        if(st) DIE("CPXbaropt(stageB) failed (%d)", st);
    }

    free(x_qp);

    int stat = CPXgetstat(env, lp);
    char stbuf[512]={0};
    CPXgetstatstring(env, stat, stbuf);
    printf("[CPLEX] status code = %d | %s\n", stat, stbuf);

    const int NC = NE;
    double* x = (double*)malloc(NC*sizeof(double));
    if(!x) DIE("oom x");
    st = CPXgetx(env, lp, x, 0, NC-1);
    if(st) DIE("CPXgetx failed (%d)", st);

    double maxRowErr=0.0;
    for(int r=0;r<nRows;r++){
        int d=rows[r].nSup; double s=0.0;
        for(int t=0;t<d;t++) s += x[ rows[r].rowOff + t ];
        double err=fabs(s-1.0); if(err>maxRowErr) maxRowErr=err;
    }
    printf("[Diag] max |sum_j a_ij - 1| = %.3e\n", maxRowErr);

    {
        double mean_self=0.0, sq_self=0.0;
        for(int e=0;e<NE;e++){
            if(eBuyer[e]==eSeller[e]){
                double a = x[e]; mean_self += a; sq_self += a*a;
            }
        }
        printf("[Diag] (1/N_F) sum a_ii = %.6e (cap %.6e)\n",
               (N_F>0? mean_self/N_F:0.0), eta1);
        printf("[Diag] (1/N_F) sum a_ii^2 = %.6e (cap %.6e)\n",
               (N_F>0? sq_self/N_F:0.0), eta2);
    }

    {
        FILE* fw=fopen(FOUT,"w");
        if(!fw) DIE("open %s", FOUT);
        fprintf(fw,"buyer,seller,weight\n");
        for(int e=0;e<NE;e++){
            int i=eBuyer[e], j=eSeller[e];
            fprintf(fw,"%d,%d,%.17g\n", i, j, x[e]);
        }
        fclose(fw);
        printf("Wrote %s\n", FOUT);
    }

    free(x);
    free(inNet);

    CPXfreeprob(env, &lp);
    CPXcloseCPLEX(&env);

    for(int l=0;l<S;l++) if(firmsInSec[l]) free(firmsInSec[l]);
    free(firmsInSec); free(secCnt); free(s_tot);

    for(int j=1;j<=maxID;j++) if(colind[j]) free(colind[j]);
    free(colind); free(colcnt); free(colpos);

    free(eBuyer); free(eSeller);
    for(int r=0;r<nRows;r++) free(rows[r].sup);
    free(rows);

    free(m); free(delta);
    free(firm2sec); free(sector_ids);

    for(int l=0;l<S;l++) if(secEdges && secEdges[l]) free(secEdges[l]);
    free(secEdges); free(secEdgeCnt); free(secFill);
    free(sellerSecPos);

    return 0;
}
