function [Nodes3D,Mesh3D] = Mesh2D_to_Mesh3D(Nodes,Mesh2D,zz)

n=size(Nodes,1);
Nz=length(zz); 

Nodes3D=[];
Mesh3D=[];

for i=1:1:Nz-1
    
Nodes3D=[Nodes3D; [Nodes zz(i)*ones(n,1)]];   

Mesh3D=[Mesh3D; [Mesh2D+(i-1)*n Mesh2D+i*n]];
    
end

Nodes3D=[Nodes3D; [Nodes zz(Nz)*ones(n,1)]];   

end

