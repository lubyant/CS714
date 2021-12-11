Lx=30; Ly=30;
nx=3;ny=3;
imin =2; imax=imin+nx-1;
jmin =2; jmax=jmin+ny-1;
% Crea te mesh
x ( imin : imax+1)=linspace ( 0 , Lx , nx +1);
y ( jmin : jmax+1)=linspace ( 0 , Ly , ny +1);
xm( imin : imax )= 0.5*( x ( imin : imax)+x ( imin +1:imax + 1 ) );
ym( jmin : jmax )= 0.5*( y ( jmin : jmax)+y ( jmin +1: jmax + 1 ) );
% Crea te mesh s i z e s
dx=x ( imin+1)-x ( imin ) ;
dy=y ( jmin+1)-y ( jmin ) ;
dxi=1/dx ;
dyi=1/dy ;

%%

% Crea te L a pl ac i a n o p e r a t o r f o r s o l v i n g p r e s s u r e Po is son e q u a t i o n
L=zeros ( nx*ny , nx*ny ) ;
for j =1:ny
    for i =1:nx
        L( i +( j-1)*nx , i +( j-1)*nx)=2*dxi^2+2*dyi^2;
        for ii=i-1: 2: i +1
            if ( ii >0 && ii <=nx ) % I n t e r i o r p o i n t
                L( i +( j-1)*nx , ii +( j-1)*nx)=-dxi^2;
            else % Neuman c o n d i t i o n s on boundary
                L( i +( j-1)*nx , i +( j-1)*nx)= ...
                    L( i +( j-1)*nx , i +( j-1)*nx)-dxi^2;
            end
        end
        for jj=j-1: 2: j+1
            if ( jj >0 && jj <=ny ) % I n t e r i o r p o i n t
                L( i +( j-1)*nx , i +( jj-1)*nx)=-dyi^2;
            else % Neuman c o n d i t i o n s on boundary
                L( i +( j-1)*nx , i +( j -1)*nx)= ...
                    L( i +( j -1)*nx , i +( j -1)*nx)-dyi^2;
            end
        end
    end
end
% Se t p r e s s u r e in f i r s t c e l l ( a l l o t h e r p r e s s u r e s w . r . t t o t h i s one )
L ( 1 , : ) = 0 ; L ( 1 , 1 )= 1;
