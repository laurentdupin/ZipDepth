#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(push_constant) uniform P{uint iw;uint ih;uint ow;uint oh;uint channels;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,c=gl_GlobalInvocationID.z;if(x>=p.ow||y>=p.oh||c>=p.channels)return;uint sx=min(p.iw-1,x*p.iw/p.ow),sy=min(p.ih-1,y*p.ih/p.oh);o.d[(c*p.oh+y)*p.ow+x]=i.d[(c*p.ih+sy)*p.iw+sx];}
