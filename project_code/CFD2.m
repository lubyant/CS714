Lx=30; Ly=30;
nx=100;ny=100;
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
