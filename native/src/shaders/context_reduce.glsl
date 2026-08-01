#version 450
layout(local_size_x=64) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(binding=2,std430) readonly buffer L{float d[];}l;
layout(push_constant) uniform P{uint channels;uint plane;}p;
void main(){uint c=gl_GlobalInvocationID.x;if(c>=p.channels)return;float m=-3.402823e38;for(uint x=0;x<p.plane;++x)m=max(m,l.d[x]);float den=0,num=0;for(uint x=0;x<p.plane;++x){float w=exp(l.d[x]-m);den+=w;num+=w*i.d[c*p.plane+x];}o.d[c]=num/den;}
