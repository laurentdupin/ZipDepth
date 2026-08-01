#version 450
layout(local_size_x=64) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(push_constant) uniform P{uint channels;uint plane;}p;
void main(){uint c=gl_GlobalInvocationID.x;if(c>=p.channels)return;float s=0;for(uint x=0;x<p.plane;++x)s+=i.d[c*p.plane+x];o.d[c]=s/float(p.plane);}
