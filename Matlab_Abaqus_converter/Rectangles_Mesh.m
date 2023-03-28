function [Nodes, Rectangles]=Rectangles_Mesh(xx,yy)
Nx=length(xx);
Ny=length(yy);

for j=1:1:Ny
for i=1:1:Nx
c=Nx*(j-1)+i;    
Nodes(c,1)=xx(i);
Nodes(c,2)=yy(j);    
end
end

for j=1:1:Ny-1
for i=1:1:Nx-1
d=(Nx-1)*(j-1)+i;    
Rectangles(d,1)=Nx*(j-1)+i;
Rectangles(d,2)=Nx*(j-1)+i+1;
Rectangles(d,3)=Nx*j+i+1;
Rectangles(d,4)=Nx*j+i;
end
end

end