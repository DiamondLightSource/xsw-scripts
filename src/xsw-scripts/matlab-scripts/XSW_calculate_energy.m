function energy = XSW_calculate_energy(lps0,hkl,Bragg_angle)

plane1 = hkl;



h = 6.63e-34;
cspeed = 3e8;
eJ = 1.6e-19;
ang = 1e-10;



lps=lps0;lps(1,4:6)=lps(1,4:6)*pi/180;
ucvs=lps(1)*lps(2)*lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6))); %Unit cell volume in A^3, sample.


lvs=[lps(1)             ,0                                                          ,0;
    lps(2)*cos(lps(6))  ,lps(2)*sin(lps(6))                                         ,0;
    lps(3)*cos(lps(5))  ,lps(3)*(cos(lps(4))-cos(lps(5))*cos(lps(6)))/sin(lps(6))   ,lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6)))/sin(lps(6))];%Real space lattice vectors a, b, and c in Cartesian coordinates with a parallel to X and b in the XY plane

rlvs=[lvs(2,2)*lvs(3,3)-lvs(2,3)*lvs(3,2)   ,lvs(2,3)*lvs(3,1)-lvs(2,1)*lvs(3,3)    ,lvs(2,1)*lvs(3,2)-lvs(2,2)*lvs(3,1);
      lvs(3,2)*lvs(1,3)-lvs(3,3)*lvs(1,2)   ,lvs(3,3)*lvs(1,1)-lvs(3,1)*lvs(1,3)    ,lvs(3,1)*lvs(1,2)-lvs(3,2)*lvs(1,1);
      0                                     ,0                                      ,lps(1)*lps(2)*sin(lps(6))]/ucvs;


% ucv=lc(1)*lc(2)*lc(3)*sin(lc(5)*pi/180);
% abc=[lc(1),0,0;0,lc(2),0;lc(3)*cos(lc(5)*pi/180),0,lc(3)*sin(lc(5)*pi/180)];
% rlv(1,:)=(lc(2)*lc(3)/ucv)*[sin(lc(5)*pi/180),0,-cos(lc(5)*pi/180)];
% rlv(2,:)=(lc(1)*lc(3)*sin(lc(5)*pi/180)/ucv)*[0,1,0];
% rlv(3,:)=(lc(1)*lc(2)/ucv)*[0,0,1];

rlv = plane1*rlvs;

dhkl=sqrt(sum((rlv).^2)).^(-1);%1/sqrt(sum(abs(plane2*rlvs).^2))

lam = 2*dhkl*ang*sind(Bragg_angle);
energy = h*cspeed/lam/eJ;


