#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(push_constant) uniform P{uint iw;uint ih;uint ow;uint oh;uint channels;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,c=gl_GlobalInvocationID.z;if(x>=p.ow||y>=p.oh||c>=p.channels)return;uint x0=x*p.iw/p.ow,x1=((x+1)*p.iw+p.ow-1)/p.ow,y0=y*p.ih/p.oh,y1=((y+1)*p.ih+p.oh-1)/p.oh;float s=0;for(uint yy=y0;yy<y1;++yy)for(uint xx=x0;xx<x1;++xx)s+=i.d[(c*p.ih+yy)*p.iw+xx];o.d[(c*p.oh+y)*p.ow+x]=s/float((x1-x0)*(y1-y0));}
