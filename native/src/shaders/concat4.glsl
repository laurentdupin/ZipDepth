#version 450
layout(local_size_x=256) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer A{float d[];}a;
layout(binding=2,std430) readonly buffer B{float d[];}b;
layout(binding=3,std430) readonly buffer C{float d[];}c;
layout(binding=4,std430) readonly buffer D{float d[];}d;
layout(push_constant) uniform P{uint each_count;}p;
void main(){uint x=gl_GlobalInvocationID.x;if(x>=p.each_count*4)return;uint q=x/p.each_count,r=x%p.each_count;o.d[x]=q==0?a.d[r]:q==1?b.d[r]:q==2?c.d[r]:d.d[r];}
