clear;
clc;
Lx=1; Ly=1;
nx=3;ny=3;nu = 0.01;
imin =2; imax=imin+nx-1;
jmin =2; jmax=jmin+ny-1;
dt = 0.001;rho = 1;
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
v = zeros(imax+1,jmax+1); u = zeros(imax+1,jmax+1);us = zeros(imax+1,jmax+1);vs = zeros(imax+1,jmax+1);p=zeros ( imax , jmax ) ;
%%
for t = 1:1/dt

u ( : , jmin -1)=2*0 - u ( : , jmin ) ;
u(imin,:)=0;
u(imax+1,:)=0;
u ( : , jmax+1)=2* 1- u ( : , jmax ) ;
v ( imin -1 ,:)=2* 0 - v ( imin , : ) ;
v ( imax +1 ,:)=2* 0 - v ( imax , : ) ;
v(:,jmin) = 0;
v(:,jmax+1) = 0;
%%
for j=jmin : jmax
    for i=imin +1:imax
        vhere =0.25*( v ( i -1, j )+v ( i -1, j +1)+v ( i , j )+v ( i , j + 1 ) );
        us ( i , j )=u ( i , j )+dt * ...
            ( nu*( u ( i -1, j )-2*u ( i , j )+u ( i +1, j ) ) * dxi^2 ...
            +nu*( u ( i , j-1)-2*u ( i , j )+u ( i , j +1))* dyi^2 ...
            -u ( i , j ) * ( u ( i +1, j )-u ( i -1, j ) ) * 0.5 * dxi ...
            -vhere *( u ( i , j +1)-u ( i , j -1 ) )*0.5* dyi ) ;
    end
end

for j=jmin +1: jmax
    for i=imin:imax
        uhere =0.25*( u ( i, j-1 )+u ( i , j )+u ( i+1 , j-1 )+u ( i+1 , j ) );
        vs ( i , j )=v ( i , j )+dt * ...
            ( nu*( v ( i -1, j )-2*v ( i , j )+v ( i +1, j ) ) * dxi^2 ...
            +nu*( v ( i , j -1)-2*v ( i , j )+v ( i , j +1))* dyi^2 ...
            -uhere * ( v ( i +1, j )-v ( i -1, j ) ) * 0.5 * dxi ...
            -v(i,j) *( v ( i , j +1)-v ( i , j -1 ) )*0.5* dyi ) ;
    end
end
%%

% Crea te L a pl ac i a n o p e r a t o r f o r s o l v i n g p r e s s u r e Po is son e q u a t i o n
L=zeros ( nx*ny , nx*ny ) ;
for j =1:ny
    for i =1:nx
        L( i +( j-1)*nx , i +( j-1)*nx)=2*dxi^2+2*dyi^2;
        for ii=i-1: 2: i +1
            if ( ii >0 && ii <=nx ) % I n t e r i o r p o i n t
                L( i +( j-1)*nx , ii +(j-1)*nx)=-dxi^2;
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

n=0;
for j=jmin : jmax
    for i=imin : imax
        n=n+1;
        R( n)=-rho / dt * ...
            ( ( us ( i +1, j )-us ( i , j ) ) *dxi ...
        +(vs ( i , j +1)-vs ( i , j ) ) * dyi ) ;
    end
end
% R(1) = 0;
R = reshape(R,[],1);

pv = L\R;

n=0;
p_k=zeros ( imax , jmax ) ;
for j=jmin : jmax
    for i=imin : imax
        n=n+1;
        p_k ( i , j )=pv ( n ) ;
    end
end

p = p + p_k;

for j=jmin : jmax
    for i=imin +1:imax
        u ( i , j )=us ( i , j )-dt / rho *( p ( i , j )-p ( i -1, j ) ) * dxi ;
    end
end
for j=jmin +1: jmax
    for i=imin : imax
        v ( i , j )=vs ( i , j )-dt / rho *( p ( i , j )-p ( i , j -1))* dyi ;
    end
end

end
figure()
contourf(u(imin:imax,jmin:jmax))
colorbar;
figure()
m = sqrt(u.^2+v.^2);
contourf(m)
figure()
contourf(p(imin:imax,jmin:jmax))
colorbar