function Plot_Mesh3D(Nodes3D,Bricks)

figure;

Rectangles=[];
Rectangles=[Rectangles ; [Bricks(:,1) Bricks(:,2) Bricks(:,3)  Bricks(:,4)]];
Rectangles=[Rectangles ; [Bricks(:,5) Bricks(:,6) Bricks(:,7)  Bricks(:,8)]];
Rectangles=[Rectangles ; [Bricks(:,1) Bricks(:,2) Bricks(:,6)  Bricks(:,5)]];
Rectangles=[Rectangles ; [Bricks(:,2) Bricks(:,3) Bricks(:,7)  Bricks(:,6)]];
Rectangles=[Rectangles ; [Bricks(:,3) Bricks(:,4) Bricks(:,8)  Bricks(:,7)]];
Rectangles=[Rectangles ; [Bricks(:,4) Bricks(:,1) Bricks(:,5)  Bricks(:,8)]];

patch('Faces',Rectangles,'Vertices',Nodes3D,'FaceColor','green','EdgeAlpha',1); 

daspect([1 1 1]);
view(3);

end