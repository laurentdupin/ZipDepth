#version 450
layout(local_size_x=256) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer A{float d[];}a;
layout(binding=2,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint count;uint plane;uint op;float scale;}p;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=p.count)return;float x=a.d[i],y=b.d[p.op>=2u?i/p.plane:i];if(p.op==0u)o.d[i]=x*y;else if(p.op==1u)o.d[i]=x+y*p.scale;else if(p.op==2u)o.d[i]=x*y;else if(p.op==3u)o.d[i]=x+y;else o.d[i]=1.0/(1.0+exp(-x));}
