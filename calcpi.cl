/*
double calc_pi(int n)
{
	int i;
	double x, area, pi;

	area= 0.0;
	for(i=1; i<n; i++) {
		x = (i+0.5)/n;
		area += 4.0/(1.0 + x*x);
	}
	pi = area/n;

	return(pi);
}
*/
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define LOCK(a) atom_cmpxchg(a, 0, 1)
#define UNLOCK(a) atom_xchg(a, 0)

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void atom_add_double(__global double *val, double delta)
{
  union {
  double f;
  ulong  i;
  } old, new;

  do
  {
   old.f = *val;
   new.f = old.f + delta;
  } 
  while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);

}

//#pragma OPENCL EXTENSION cl_intel_printf : enable
__kernel void calc_pi(__global double *area,
						__global int *mutex,
						int n){
	int i = 0;
	double x = 0.0;
	i = get_global_id(0);
	if(i >= 0 && i < n){
		x = (i + 0.5)/n;
		double aux = 4.0/(1.0 + x*x);
		//while(LOCK(mutex));
		atom_add_double(area, aux);
        //UNLOCK(mutex);
	}  
}