#version 450
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer D{float d[];}d;
layout(binding=2,std430) readonly buffer W{float d[];}w;
layout(push_constant) uniform P{uint width;uint height;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,ow=p.width*2,oh=p.height*2;if(x>=ow||y>=oh)return;uint px=x/2,py=y/2,s=(y%2)*2+x%2;float m=-3.402823e38;for(uint n=0;n<9;++n)m=max(m,w.d[((n*4+s)*p.height+py)*p.width+px]);float den=0,v=0;for(uint n=0;n<9;++n){float a=exp(w.d[((n*4+s)*p.height+py)*p.width+px]-m);den+=a;uint sx=uint(clamp(int(px)+int(n%3)-1,0,int(p.width)-1)),sy=uint(clamp(int(py)+int(n/3)-1,0,int(p.height)-1));v+=a*d.d[sy*p.width+sx];}o.d[y*ow+x]=max(v/den,0.0);}
