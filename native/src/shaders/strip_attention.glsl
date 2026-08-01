#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(binding=2,std430) readonly buffer W{float d[];}w;
layout(binding=3,std430) readonly buffer G{float d[];}g;
layout(binding=4,std430) readonly buffer B{float d[];}b;
layout(binding=5,std430) readonly buffer M{float d[];}m;
layout(binding=6,std430) readonly buffer V{float d[];}v;
layout(push_constant) uniform P{uint width;uint height;uint channels;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,c=gl_GlobalInvocationID.z;if(x>=p.width||y>=p.height||c>=p.channels)return;uint plane=p.width*p.height;float hs=0,ws=0;for(uint xx=0;xx<p.width;++xx)hs+=i.d[c*plane+y*p.width+xx];for(uint yy=0;yy<p.height;++yy)ws+=i.d[c*plane+yy*p.width+x];float z=(hs/float(p.width)+ws/float(p.height))*w.d[c];z=(z-m.d[c])*g.d[c]*inversesqrt(v.d[c]+1e-5)+b.d[c];float gate=1.0/(1.0+exp(-z));uint q=c*plane+y*p.width+x;o.d[q]=i.d[q]*gate;}
