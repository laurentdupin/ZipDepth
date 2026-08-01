#version 450
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,std430) writeonly buffer O{float d[];}o;
layout(binding=1,std430) readonly buffer D{float d[];}d;
layout(binding=2,std430) readonly buffer A{float d[];}a;
layout(push_constant) uniform P{uint width;uint height;}p;
float sample_bi(uint x,uint y){uint ow=p.width*2,oh=p.height*2;float sx=clamp((float(x)+.5)*p.width/ow-.5,0.,float(p.width-1)),sy=clamp((float(y)+.5)*p.height/oh-.5,0.,float(p.height-1));uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,p.width-1),y1=min(y0+1,p.height-1);return mix(mix(d.d[y0*p.width+x0],d.d[y0*p.width+x1],sx-x0),mix(d.d[y1*p.width+x0],d.d[y1*p.width+x1],sx-x0),sy-y0);}
float alpha_bi(uint x,uint y){uint ow=p.width*2,oh=p.height*2;float sx=clamp((float(x)+.5)*p.width/ow-.5,0.,float(p.width-1)),sy=clamp((float(y)+.5)*p.height/oh-.5,0.,float(p.height-1));uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,p.width-1),y1=min(y0+1,p.height-1);return mix(mix(a.d[y0*p.width+x0],a.d[y0*p.width+x1],sx-x0),mix(a.d[y1*p.width+x0],a.d[y1*p.width+x1],sx-x0),sy-y0);}
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,ow=p.width*2,oh=p.height*2;if(x>=ow||y>=oh)return;float q=1.0/(1.0+exp(-alpha_bi(x,y))),nn=d.d[(y/2)*p.width+x/2],bi=sample_bi(x,y);o.d[y*ow+x]=max(q*nn+(1-q)*bi,0.0);}
