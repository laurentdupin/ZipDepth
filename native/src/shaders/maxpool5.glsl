#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(push_constant) uniform P{uint width;uint height;uint channels;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,c=gl_GlobalInvocationID.z;if(x>=p.width||y>=p.height||c>=p.channels)return;float m=-3.402823e38;for(int yy=-2;yy<=2;++yy)for(int xx=-2;xx<=2;++xx){uint sx=uint(clamp(int(x)+xx,0,int(p.width)-1)),sy=uint(clamp(int(y)+yy,0,int(p.height)-1));m=max(m,i.d[(c*p.height+sy)*p.width+sx]);}o.d[(c*p.height+y)*p.width+x]=m;}
