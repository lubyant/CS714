%-----------------------------------------------------------------------
% CFD project 2
% Boyuan Lu
%-----------------------------------------------------------------------
% set up constant variables
Lx=30; Ly=30; x0=0.5*Lx; y0=0.5*Ly;
garma=1;sigma=2.5; vis=0.001; rho=1; dt=0.01;T=1;nt= T/dt+1;
nx=100;ny=100;dx=Lx/(nx-1);dy=Ly/(ny-1);
u=zeros(nx,ny);v=zeros(nx,ny);
u_i=zeros(nx,ny);v_i=zeros(nx,ny);
u_s=zeros(nx,ny);v_s=zeros(nx,ny);
uf_s=zeros(nx,ny);vf_s=zeros(nx,ny);
R=zeros(nx,ny);P = ones(nx,ny);
niu = 0.001;rou = 1;
u = ones(nx,ny);
v = zeros(nx,ny);
uf = zeros(nx,ny);
vf = zeros(nx,ny);
% set up initial condition
x=0:dx:Lx;
y=0:dy:Ly;
for i = 1:nx
    for j = 1:ny
        u(i,j) =1-4/25*(y(j)-15)*exp(2/25*(-(y(j)-15).^2-(x(i)-15).^2));
        v(i,j) = 4/25*(x(i)-15)*exp(2/25*(-(x(i)-15).^2-(y(j)-15).^2));
    end
end
P_new = ones(nx,ny);
for t = 2:nt
    [Hx,Hy,uf,vf]=HCD(u,v,nx,ny);
    for i= 1:nx
        for j = 1:ny
            if i==1
                ui(i,j)=u(i,j)+dt*Hx(i,j)-dt/rou*(P(i+1,j)-P(i,j))/(dx);
            elseif i==nx
                ui(i,j)=u(i,j)+dt*Hx(i,j)-dt/rou*(P(i,j)-P(i-1,j))/(dx);
            else
                ui(i,j)=u(i,j)+dt*Hx(i,j)-dt/rou*(P(i+1,j)-P(i-1,j))/(2*dx);
            end
            if j==1
                vi(i,j)=v(i,j)+dt*Hy(i,j)-dt/rou*(P(i,j+1)-P(i,j))/(dy);
            elseif j==ny
                vi(i,j)=v(i,j)+dt*Hy(i,j)-dt/rou*(P(i,j)-P(i,j-1))/(dy);
            else
                vi(i,j)=v(i,j)+dt*Hy(i,j)-dt/rou*(P(i,j+1)-P(i,j-1))/(2*dy);
            end
        end
    end
    [Hxi,Hyi]=HCD(ui,vi,nx,ny);
    for i=1:nx
        for j = 1:ny
            if i==1
                u_s(i,j)=u(i,j)+dt/2*(Hx(i,j)+Hxi(i,j))-dt/rou*(P(i+1,j)-P(i,j))/(dx);
            elseif i==nx
                u_s(i,j)=u(i,j)+dt/2*(Hx(i,j)+Hxi(i,j))-dt/rou*(P(i,j)-P(i-1,j))/(dx);
            else
                u_s(i,j) = u(i,j) + dt/2*(Hxi(i,j) + Hx(i,j)) - dt/rou*(P(i+1,j)-P(i-1,j))/(2*dx);
            end
            if j==1
                v_s(i,j)=v(i,j)+dt/2*(Hy(i,j)+Hyi(i,j))-dt/rou*(P(i,j+1)-P(i,j))/(dy);
            elseif j==ny
                v_s(i,j)=v(i,j)+dt/2*(Hy(i,j)+Hyi(i,j))-dt/rou*(P(i,j)-P(i,j-1))/(dy);
            else
                v_s(i,j) = v(i,j) + dt/2*(Hyi(i,j) + Hy(i,j)) - dt/rou*(P(i,j+1)-P(i,j-1))/(2*dy);
            end
        end
    end
    %   uf_s, vf_s
    for i = 1:nx-1
        for j=1:ny-1
            uf_s(i,j) = 0.5*(u_s(i+1,j)+u_s(i,j));
            vf_s(i,j) = 0.5*(v_s(i,j+1)+v_s(i,j));
        end
    end
    uf_s(nx,:)=u_s(nx,:);
    vf_s(nx,:)=v_s(nx,:);
    uf_s(:,ny)=u_s(:,ny);
    vf_s(:,ny)=v_s(:,ny);
    
    for i=2:nx-1
        for j=2:ny-1
            if i==2&&j==2
                R(i,j)=2*rou/dt*((uf_s(i,j)-1)/dx+(vf_s(i,j)-0)/dy);
            elseif i==2&&j==ny-1
                R(i,j)=2*rou/dt*((uf_s(i,j)-1)/dx+(0-vf_s(i,j-1))/dy);
            elseif i==nx-1&&j==2
                R(i,j)=2*rou/dt*((1-uf_s(i-1,j))/dx+(vf_s(i,j)-0)/dy);
            elseif i==nx-1&&j==ny-1
                R(i,j)=2*rou/dt*((1-uf_s(i-1,j))/dx+(vf_s(i,j)-vf_s(i,j-1))/dy);
            elseif i==2&&j~=2&&j~=ny-1
                R(i,j)=2*rou/dt*((uf_s(i+1,j)-1)/dx+(vf_s(i,j)-vf_s(i,j-1))/dy);
            elseif j==2&&i~=2&&i~=ny-1
                R(i,j)=2*rou/dt*((uf_s(i,j)-uf_s(i-1,j))/dx+(vf_s(i,j+1)-0)/dy);
            elseif i==nx-1&&j~=2&&j~=ny-1
                R(i,j)=2*rou/dt*((1-uf_s(i-1,j))/dx+(vf_s(i,j)-vf_s(i,j-1))/dy);
            elseif j==ny-1&&i~=2&&i~=nx-1
                R(i,j)=2*rou/dt*((uf_s(i,j)-uf_s(i-1,j))/dx+(0-vf_s(i,j-1))/dy);
            else
                R(i,j)=2*rou/dt*((uf_s(i,j)-uf_s(i-1,j))/dx+(vf_s(i,j)-vf_s(i,j-1))/dy);
            end
        end
    end
    
    %     ROS
    P_old=P;
    error=1;
    P_new=zeros(nx,ny);
    while(error>1e-3)
        for j=2:ny-1
            for i=2:nx-1
                if i==2&&j==2
                    P_new(i,j)=(P_new(i+1,j)*dy*dy+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+dy*dy);
                elseif i==2&&j==ny-1
                    P_new(i,j)=(P_new(i+1,j)*dy*dy+P_new(i,j-1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+dy*dy);
                elseif i==nx-1&&j==2
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+dy*dy);
                elseif i==nx-1&&j==ny-1
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i,j-1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+dy*dy);
                elseif i==2&&j~=2&&j~=ny-1
                    P_new(i,j)=(P_new(i+1,j)*dy*dy+P_new(i,j-1)*dx*dx+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(2*dx*dx+dy*dy);
                elseif i==nx-1&&j~=2&&j~=ny-1
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i,j-1)*dx*dx+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(2*dx*dx+dy*dy);
                elseif j==2&&i~=2&&i~=nx-1
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i+1,j)*dy*dy+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+2*dy*dy);
                elseif j==ny-1&&i~=2&&i~=nx-1
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i+1,j)*dy*dy+P_new(i,j-1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(dx*dx+2*dy*dy);
                else
                    P_new(i,j)=(P_new(i-1,j)*dy*dy+P_new(i+1,j)*dy*dy+P_new(i,j-1)*dx*dx+P_new(i,j+1)*dx*dx-R(i,j)*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy);
                end
            end
        end
        P_new(1,:)=P_new(2,:);
        P_new(nx,:)=P_new(nx-1,:);
        P_new(:,1)=P_new(:,2);
        P_new(:,nx)=P_new(:,nx-1);
        error1=max(abs(P_new-P_old));
        error=max(error1);
        P_old=P_new;
    end
    % update u(n+1),v(n+1)
    for i=1:nx
        for j = 1:ny
            if i==1
                u(i,j)=u_s(i,j)-dt/(2*rou)*(P_new(i+1,j)-P_new(i,j))/(dx);
            elseif i==nx
                u(i,j)=u_s(i,j)-dt/(2*rou)*(P_new(i,j)-P_new(i-1,j))/(dx);
            else
                u(i,j) = u_s(i,j) - dt/(2*rou)*(P_new(i+1,j)-P_new(i-1,j))/(2*dx);
            end
            if j==1
                v(i,j)=v_s(i,j)-dt/(2*rou)*(P_new(i,j+1)-P_new(i,j))/(dy);
            elseif j==ny
                v(i,j)=v_s(i,j)-dt/(2*rou)*(P_new(i,j)-P_new(i,j-1))/(dy);
            else
                v(i,j) = v_s(i,j)- dt/(2*rou)*(P_new(i,j+1)-P_new(i,j-1))/(2*dy);
            end
        end
    end
    %     update P
    P=P+P_new;
end
% plot the figures
figure(1);
surface(u);
view(50,30);title('u')
figure(2);
surface(v);
view(50,30);title('v')
figure(3);
surface(P);
view(50,30);title('pressure')
figure(4);
surf(divergence(-v,-u));
view(50,30);title('divergence u v')
figure(5);
surf(-curl(v,u));
view(50,30);title('vort u v')
figure(6)
surf(divergence(vf,uf));
view(50,30);title('divergence uf vf')

%%
% This is the function to generate H terms

function [H_x,H_y,uf_w,vf_w]=HCD(u,v,nx,ny)
Lx=30; Ly=30;dt=0.01;vis=0.001;
dx=Lx/(nx-1);dy=Ly/(ny-1);
x=0:dx:Lx; y=0:dy:Ly;
for j=1:ny
    for i=1:nx
        if j==1
            u(i,j)=u(i,j+1);v(i,j)=0;
            uf(i,j)=u(i,j);vf(i,j)=0;
        elseif j==ny
            u(i,j)=u(i,j-1);v(i,j)=0;
            uf(i,j)=u(i,j);vf(i,j)=0;
        elseif i==1
            u(i,j)=1; v(i,j)=0;
            uf(i,j)=1; vf(i,j)=0;
        elseif i==nx
            u(i,j)=1; v(i,j)=0;
            uf(i,j)=1; vf(i,j)=0;
        else
            uf(i,j)=0.5*(u(i+1,j)+u(i,j));
            vf(i,j)=0.5*(v(i,j+1)+v(i,j));
        end
    end
end
c=mean(uf(nx,:));
uf_w=uf;vf_w=vf;
uf_w(nx,:)=uf(nx,:)+c*(uf(nx,:)-uf(nx-1,:))*dt/dx;
vf(nx,:)=vf(nx,:)+c*(vf(nx,:)-vf(nx-1,:))*dt/dx;
epi=dy/Ly*(sum(uf(1,:)-uf(nx,:)));
uf(nx,:)=uf_w(nx,:)+epi;
% form the H terms
C_x=zeros(nx,ny);C_y=zeros(nx,ny);D_x=zeros(nx,ny);D_y=zeros(nx,ny);
for j=2:ny
    for i=2:nx
        C_x(i,j)=(uf(i,j)^2-uf(i-1,j)^2)/dx+(uf(i,j)*vf(i,j)-uf(i,j-1)*vf(i,j-1))/dy;
        C_y(i,j)=(uf(i,j)*vf(i,j)-uf(i-1,j)*vf(i-1,j))/dx+(vf(i,j)^2-vf(i,j-1)^2)/dy;
    end
end
for j=1:ny
    for i=1:nx
        if i==1&&j==1
            D_x(i,j)=vis*(2*u(i+1,j)-2*u(i,j))/(dx^2)+vis*(2*u(i,j+1)-2*u(i,j))/(dy^2);
            D_y(i,j)=vis*(2*v(i+1,j)-2*v(i,j))/(dx^2)+vis*(2*v(i,j+1)-2*v(i,j))/(dy^2);
        elseif i==1&&j==ny
            D_x(i,j)=vis*(2*u(i+1,j)-2*u(i,j))/(dx^2)+vis*(-2*u(i,j)+2*u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(2*v(i+1,j)-2*v(i,j))/(dx^2)+vis*(-2*v(i,j)+2*v(i,j-1))/(dy^2);
        elseif i==nx&&j==1
            D_x(i,j)=vis*(-2*u(i,j)+2*u(i-1,j))/(dx^2)+vis*(2*u(i,j+1)-2*u(i,j))/(dy^2);
            D_y(i,j)=vis*(-2*v(i,j)+2*v(i-1,j))/(dx^2)+vis*(2*v(i,j+1)-2*v(i,j))/(dy^2);
        elseif i==nx&&j==ny
            D_x(i,j)=vis*(-2*u(i,j)+2*u(i-1,j))/(dx^2)+vis*(-2*u(i,j)+2*u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(-2*v(i,j)+2*v(i-1,j))/(dx^2)+vis*(-2*v(i,j)+2*v(i,j-1))/(dy^2);
        elseif i==1&&j~=1&&j~=ny
            D_x(i,j)=vis*(2*u(i+1,j)-2*u(i,j))/(dx^2)+vis*(u(i,j+1)-2*u(i,j)+u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(2*v(i+1,j)-2*v(i,j))/(dx^2)+vis*(v(i,j+1)-2*v(i,j)+v(i,j-1))/(dy^2);
        elseif i==nx&&j~=1&&j~=ny
            D_x(i,j)=vis*(-2*u(i,j)+2*u(i-1,j))/(dx^2)+vis*(u(i,j+1)-2*u(i,j)+u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(-2*v(i,j)+2*v(i-1,j))/(dx^2)+vis*(v(i,j+1)-2*v(i,j)+v(i,j-1))/(dy^2);
        elseif j==1&&i~=1&&i~=nx
            D_x(i,j)=vis*(u(i+1,j)-2*u(i,j)+u(i-1,j))/(dx^2)+vis*(2*u(i,j+1)-2*u(i,j))/(dy^2);
            D_y(i,j)=vis*(v(i+1,j)-2*v(i,j)+v(i-1,j))/(dx^2)+vis*(2*v(i,j+1)-2*v(i,j))/(dy^2);
        elseif j==ny&&i~=1&&i~=nx
            D_x(i,j)=vis*(u(i+1,j)-2*u(i,j)+u(i-1,j))/(dx^2)+vis*(-2*u(i,j)+2*u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(v(i+1,j)-2*v(i,j)+v(i-1,j))/(dx^2)+vis*(-2*v(i,j)+2*v(i,j-1))/(dy^2);
        else
            D_x(i,j)=vis*(u(i+1,j)-2*u(i,j)+u(i-1,j))/(dx^2)+vis*(u(i,j+1)-2*u(i,j)+u(i,j-1))/(dy^2);
            D_y(i,j)=vis*(v(i+1,j)-2*v(i,j)+v(i-1,j))/(dx^2)+vis*(v(i,j+1)-2*v(i,j)+v(i,j-1))/(dy^2);
        end
    end
end
H_x=D_x-C_x; H_y=D_y-C_y;
end
