// The file cache of one file: filecache FILE core|evict|read|touch|lock|will OFF_GB LEN_GB [THREADS]
// core: how much of it the cache holds; evict: drop it from the cache (a cold start); read: bring it in.
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mach/mach_time.h>
static double now(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }
typedef struct { unsigned char *p; size_t n; unsigned long sum; } job;
static void *touch(void *a){ job *j=a; unsigned long s=0; for(size_t i=0;i<j->n;i+=16384) s+=j->p[i]; j->sum=s; return 0; }
int main(int argc,char**argv){
  int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st);
  const char *mode=argv[2]; size_t off=(size_t)(atof(argv[3])*(1ul<<30))/16384*16384, len=(size_t)(atof(argv[4])*(1ul<<30)); if(off+len>(size_t)st.st_size) len=(st.st_size-off)/16384*16384; int nt=argc>5?atoi(argv[5]):1;
  double t=now();
  if(mode[0]=='r'){ char *b=malloc(64<<20); size_t d=0; while(d<len){ ssize_t r=pread(fd,b,64<<20,off+d); if(r<=0)break; d+=r; } }
  else if(mode[0]=='a'){ struct radvisory ra={off,(int)(len>2000000000?2000000000:len)}; if(fcntl(fd,F_RDADVISE,&ra)) perror("rdadvise"); }
  else { unsigned char *p=mmap(0,len,PROT_READ,MAP_PRIVATE,fd,off);
    if(mode[0]=='l'){ if(mlock(p,len)) perror("mlock"); }
    else if(mode[0]=='e'){ munmap(p,len); p=mmap(0,len,PROT_READ,MAP_SHARED,fd,off); if(msync(p,len,MS_INVALIDATE)) perror("msync"); }
    else if(mode[0]=='c'){ char *v=malloc(len/16384+1); mincore(p,len,v); size_t in=0; for(size_t i=0;i<len/16384;i++) in+=v[i]&1; printf("in core %.2f GB of %.2f\n",in*16384.0/(1ul<<30),len/(double)(1ul<<30)); }
    else if(mode[0]=='w'){ madvise(p,len,MADV_WILLNEED); job j={p,len,0}; touch(&j); }
    else { pthread_t th[16]; job js[16]; size_t per=len/nt/16384*16384; for(int i=0;i<nt;i++){ js[i]=(job){p+i*per,per,0}; pthread_create(&th[i],0,touch,&js[i]); } for(int i=0;i<nt;i++) pthread_join(th[i],0); }
  }
  double ms=now()-t; printf("%s off %s GB len %.2f GB threads %d: %.0f ms, %.2f s/GB\n",mode,argv[3],len/(double)(1ul<<30),nt,ms,ms/1e3/(len/(double)(1ul<<30))); return 0; }
