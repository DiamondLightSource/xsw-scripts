function [fit_out] = XSW_predict_modulation(datadir,fh0,ph0,theta,plotter,varargin)
%%% calcuates XSW absorption profile from inputted parameters
% [fit_out] = XSW_predict_modulation(datadir,dataprefix,fh0,ph0,theta,plotter,varargin)
% datadir - folder containing the structure factors (*.nff files) and f0_all_free_atoms.txt
% fh0 - coherent fraction
% ph0 - coherent position
% theta - angle between the measured emission direction and the photon
%       polarisation (for most I09 stuff this is 18)
% plotter - if greater than zero plots figures and outputs text to the
%       command line
% varargin - optional variables:
%       varargin{1} = Z value of the emitter atom (e.g. 8 for oxygen)
%       varargin{2} = principle quantum number, n, for photoelectron
%                   emitting orbital
%       varargin{3} = azimuthal quantum number, l, for photoelectron
%                   emitting orbital, e.g. 0 for s, 1 for l, 2 for d, 3 for f
%       varargin{4} = total spin, js = l+ms, for photoelectron emitting orbital
%                    e.g. 3/2 for the Ti 2p 3/2 level. For an s orbital set
%                    to 1/2
%       varargin{5} = angle between the scattering plane and the surface,
%                    usually 0
%       varargin{6} = (h,k,l) index of the reflection, e.g. [1,1,1] for the
%                   (111)
%       varargin{7} = lattice unit cell = [a b c, alpha beta gamma] in Å
%                       and °. e.g. for Cu:
%                   = [3.6149,3.6149,3.6149, 90,90,90]
%       varargin{8} = position of atoms in the unit cell in fractional
%                       coordinates and an occupational factor (usually 1) in the 
%                       format [Z, x, y, z, f], e.g. for fcc Cu:
%               = [29, 0.0, 0.0, 0.0, 1;
%                  29, 0.5, 0.5, 0.0, 1;
%                  29, 0.5, 0.0, 0.5, 1;
%                  29, 0.0, 0.5, 0.5, 1;]
%       varargin{9} = gaussian broadening, models the experimental
%               broadening due to imperfections in the monochromator (pretty small)
%               and the sample substrate (significantly larger). Numbers between
%               0.1 and 0.5 eV are common, 0.3 is broadly the mean




z={'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No'};



%%% Bragg angle
thetaB = 90-4;

if isempty(varargin)


    samp='Cu';  %Sample crystal
    
    for nn = 1:length(z)
        if z{nn}==samp
            atom_type = nn;
        end
    end    
    %%% angle between the scattering plane and the surface
    alphaB = 0;
    
    energy = 2971;  %Photon energy in eV
    a_lat = 3.6149;  %lattice parameter of the crystal
    
    hs=[1,1,1]; %Sample reflection
    
    lps0=[a_lat,a_lat,a_lat,90,90,90]; %lattice unit cell = [a b c, alpha beta gamma]
    xyzs=[ 
    atom_type,0,0,0,1;
    atom_type,0.5,0.5,0,1;
    atom_type,0.0,0.5,0.5,1;
    atom_type,0.5,0.0,0.5,1;];

    width = 0.2;

    [bettab,gamtab,deltab,Eb] = q_param(datadir,8,1,0,1/2,plotter);
else
    if length(varargin) == 9

        %%% angle between the scattering plane and the surface
        alphaB = varargin{5};
        
        
        
        hs= varargin{6}; %Sample reflection
        
        lps0= varargin{7}; %[a_lat,a_lat,a_lat,90,90,90]; %lattice unit cell = [a b c, alpha beta gamma]
        xyzs= varargin{8};

        width = varargin{9};

        if ischar(varargin(1))
            if length(varargin{1}) == 1
                varargin{1} = [' ' varargin{1}];
            end
            varargin{1} = find(strcmp(varargin{1},z)==1);
        end
        [bettab,gamtab,deltab,Eb] = q_param(datadir,varargin{1},varargin{2},varargin{3},varargin{4},plotter);
        energy = XSW_calculate_energy(lps0,hs,thetaB);  %Photon energy in eV
    else
        error('You must either include the Z, n, l and js of the emitter orbital, or you must edit the script to include this information')
    end
end       
       

fwhmgaus = width; %in eV, FWHM of the Gaussian to be convoluted with the sample theoretical rocking curve (to account1 for, e.g., mosaicity).
if datadir(end) ~= '\'
    datadir(end+1) = '\';
end
%// Non-dipolar corrections, assuming delta=0
%// C1=(1+Q)/(1-Q), C2=1/(1-Q), C3=phase shift
%// Row1=Ek, Row2=No correction, Row3=C1s, Row4=N1s, Row5=O1s, Row6=Fe2p3/2
%// Ref[1]: Trzhaskovskaya et al., Atomic Data and Nuclear Data Tables 77, 97 (2001)

%Q is defined as the forward/backward assymetry factor and includes gamma,
%delta, beta (delta however is set to 0 as the effect is very small !! ( ompare Ref[1]))
%%%% for Fe2p, compared to gamma, delta is one order or magnitude less big

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%non dipolar corrections in the order: photon energy, 0 ,C1s, N1s, O1s,
%Fe2p3/2 (delta term newly inserted on 14.07.2014 to deal correctly with Fe2p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




QCE =  0; 
% Analyser-beam geometry
the = theta*pi/180;      %Theta angle in FIG1 in Ref[1]
phie = 0*pi/180;      %Phi angle in FIG1 in Ref[1]

test_E = 1500:10:5000;

beta=interp1(bettab(1,:),bettab(QCE+2,:),test_E);
gama=interp1(gamtab(1,:),gamtab(QCE+2,:),test_E);
delt=interp1(deltab(1,:),deltab(QCE+2,:),test_E);

bet = beta(find(min(abs(test_E-energy+Eb(QCE+1)))==abs(test_E-energy+Eb(QCE+1))));      %interpolates beta values from table
gam = gama(find(min(abs(test_E-energy+Eb(QCE+1)))==abs(test_E-energy+Eb(QCE+1))));      %interpolates gamma values from table
del = delt(find(min(abs(test_E-energy+Eb(QCE+1)))==abs(test_E-energy+Eb(QCE+1)))); 

%Q=gam*cos(the)^2*sin(the)*cos(phie)/(1+bet*0.5*(3*cos(the)^2-1));      %old Q-value neglecting delta terms
Q=((del*sin(the)*cos(phie))+(gam*cos(phie)*sin(the)*cos(the)^2))/(1+bet*0.5*(3*cos(the)^2-1));       %new Q-value with delta terms to cope with Fe2p properly 

C1=(1+Q)/(1-Q); %C 1s at 30 deg, 2.63keV
C2=1/(1-Q);
C3=0;



hm=[1,1,1]; %Mono Si reflection

% datafile=[datadir dataprefix '.txt'];
% dataoutfile=[datadir dataprefix '_dat.txt'];
% fitoutfile=[datadir dataprefix '_fit.txt'];
% figurefile=[datadir dataprefix '.pdf'];

lambda=12398.54/energy;     %Wavelength in A

%%%%%%%%%%%%%%%%%%%%%
%%%Read in data files
%%%%%%%%%%%%%%%%%%%%%

%data=data_square;%dlmread(datafile);
%dataE=data; 
% Esi=[];
% for i=1:size(dataE,1)
%     %size dataE
%     %find(data(:,1)==dataE(i,1))
%     Esi=[Esi,find(data(:,1)==dataE(i,1))'];
% end
% datar=data(Esi,[1,3]);
% xshift=0;
% datay=data(Esi,[1:2]);
% dataerr = data(Esi,[1,4]);
% datay(:,1)=datay(:,1)+xshift;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Lattice parameters, DW factor and atomic coordinates, sample,
%monochromator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



DWBs=0.;    %Debye-Waller B factor, sample.


%atom_type = find(z==samp)

                                                                  %Sample Ag

nas=size(xyzs,1);


%Si monochromator
lpm0=[5.431,5.431,5.431,90,90,90];  %Lattice parameters, DW factor and atomic coordinates, Si monochromator
DWBm=0.; %Debye-Waller B factor, mono.

xyzm=[14,0,0,0,1; 14,0.5,0.5,0,1;14,0.5,0,0.5,1;14,0,0.5,0.5,1;14,0.25,0.25,0.25,1;14,0.75,0.75,0.25,1;14,0.75,0.25,0.75,1;14,0.25,0.75,0.75,1;]; %Atomic number,x,y,z,occupancy
                                                                                                                                                  %Mono Si

nam=size(xyzm,1);
xyzm(:,2:4)=xyzm(:,2:4)-ones(nam,1)*[0.125,0.125,0.125]; % Shift the origin to inversion center 

%Unit cell volumes, Bragg plane spacings and Bragg angles
lps=lps0;lps(1,4:6)=lps(1,4:6)*pi/180; %Change deg to rad, sample.
lpm=lpm0;lpm(1,4:6)=lpm(1,4:6)*pi/180; %Change deg to rad, mono.
ucvs=lps(1)*lps(2)*lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6))); %Unit cell volume in A^3, sample.
ucvm=lpm(1)*lpm(2)*lpm(3); %Unit cell volume in A^3, mono Si.
lvs=[lps(1),0,0;lps(2)*cos(lps(6)),lps(2)*sin(lps(6)),0;lps(3)*cos(lps(5)),lps(3)*(cos(lps(4))-cos(lps(5))*cos(lps(6)))/sin(lps(6)),lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6)))/sin(lps(6))];%Real space lattice vectors a, b, and c in Cartesian coordinates with a parallel to X and b in the XY plane
rlvs=[lvs(2,2)*lvs(3,3)-lvs(2,3)*lvs(3,2),lvs(2,3)*lvs(3,1)-lvs(2,1)*lvs(3,3),lvs(2,1)*lvs(3,2)-lvs(2,2)*lvs(3,1);lvs(3,2)*lvs(1,3)-lvs(3,3)*lvs(1,2),lvs(3,3)*lvs(1,1)-lvs(3,1)*lvs(1,3),lvs(3,1)*lvs(1,2)-lvs(3,2)*lvs(1,1);0,0,lps(1)*lps(2)*sin(lps(6))]/ucvs;

dhs=sqrt(sum((hs*rlvs).^2)).^(-1); %Bragg plane spacing for hkl reflection in A, sample.
dhm=lpm(1)/sqrt(hm*hm'); %Bragg plane spacing for hkl reflection in A, mono Si.

thbs=asin(dhs^(-1)*lambda/2); %Bragg angle in rad, sample.
thbm=asin(dhm^(-1)*lambda/2); %Bragg angle in rad, mono Si


%z=['Ag' 'Cu' 'Si'];
%Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Structure factors and chi values, sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f0=dlmread([datadir 'f0_all_free_atoms.txt']);

fps=ones(nas,1)*0;fpps=ones(nas,1)*0;
fs=ones(nas,1)*0;f0s=ones(nas,1)*0;
for i=1:nas
    %[datadir z{xyzs(i,1)} '.nff']
  fpfppdata=dlmread([datadir z{xyzs(i,1)} '.nff'],'\t',1,0);
  %fpfppdata(:,2)=fpfppdata(:,2)*1.05;
  %whos fpfppdata
  fps(i)=interp1(fpfppdata(:,1),fpfppdata(:,2),energy)-xyzs(i,1);
  fpps(i)=interp1(fpfppdata(:,1),fpfppdata(:,3),energy);
  fs(i)=interp1(f0(:,1),f0(:,xyzs(i,1)-1),0.5*dhs^(-1))+fps(i)+j*fpps(i);
  f0s(i)=xyzs(i,1)+fps(i)+j*fpps(i);
end;
%f0s
hrs=xyzs(:,2:4)*hs';
Fhs=(exp(2*pi*j*hrs')*(fs.*xyzs(:,5)))*exp(-DWBs/dhs^2/4);
Fhbs=(exp(-2*pi*j*hrs')*(fs.*xyzs(:,5)))*exp(-DWBs/dhs^2/4);
F0s=sum(f0s.*xyzs(:,5))*exp(-DWBs/dhs^2/4);


gams=2.818e-5*lambda^2/pi/ucvs;
chihs=-gams*Fhs;
chihbs=-gams*Fhbs;
chi0s=-gams*F0s;
%pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Structure factors and chi values, mono
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fpm=ones(nam,1)*0;fppm=ones(nam,1)*0;
fm=ones(nam,1)*0;f0m=ones(nam,1)*0;
for i=1:nam
  fpfppdata=dlmread([datadir z{xyzm(i,1)} '.nff'],'\t',1,0);
  %fpfppdata(:,2:3)=fpfppdata(:,2:3)*0.5;
  fpm(i)=interp1(fpfppdata(:,1),fpfppdata(:,2),energy)-xyzm(i,1);
  fppm(i)=interp1(fpfppdata(:,1),fpfppdata(:,3),energy);
  fm(i)=interp1(f0(:,1),f0(:,xyzm(i,1)-1),0.5*dhm^(-1))+fpm(i)+j*fppm(i);
  f0m(i)=xyzm(i,1)+fpm(i)+j*fppm(i);
end
hrm=xyzm(:,2:4)*hm';
Fhm=(exp(2*pi*j*hrm')*(fm.*xyzm(:,5)))*exp(-DWBm/dhm^2/4);
Fhbm=(exp(-2*pi*j*hrm')*(fm.*xyzm(:,5)))*exp(-DWBm/dhm^2/4);
F0m=sum(f0m.*xyzm(:,5))*exp(-DWBm/dhm^2/4);
	
gamm=2.818e-5*lambda^2/pi/ucvm;
chihm=-gamm*Fhm;
chihbm=-gamm*Fhbm;
chi0m=-gamm*F0m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Gaussian (normalized to integrated area)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

areagaus=1;
ngaus=1010;
%if fwhmgaus>0 
  rangegaus=20;%(max(datar(:,1))-min(datar(:,1)))*2;
  degaus=rangegaus/(ngaus-1);
  egaus=[-round((ngaus-1)/2):(ngaus-round((ngaus-1)/2)-1)]*degaus;
%else
%  degaus=0.005;
%end;

%%%%%%%%%%%%%%%%%%%%%
%Sample rocking curve
%%%%%%%%%%%%%%%%%%%%%
%df is the angle between photon incidence and the surface towards the
%detector
%di is the angle between photon incidence and the surface away from the
%detector
di = 45/180*pi;
df = pi-di;
bs=-sind(thetaB-alphaB)./sind(thetaB+alphaB);
Ps=1.0;
ewidths=energy*abs(real(chihs)*Ps)/sin(thbs)^2/sqrt(abs(bs));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These parameters can be played with
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
des1=-10*ewidths; 
des2=10*ewidths;
colour = 'g';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsteps=round((des2-des1)/degaus);
a1=[1:nsteps];
des=des1+(a1-1)*degaus;
etas=(2*bs*des*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
%whos a1
xs=0*ones(nsteps,1);
xs(find(real(etas)>=0))=sqrt(abs(bs))*(sqrt(chihs*chihbs)/chihbs)*(etas(find(real(etas)>=0))-sqrt(etas(find(real(etas)>=0)).^2-1));
xs(find(real(etas)<0))=sqrt(abs(bs))*(sqrt(chihs*chihbs)/chihbs)*(etas(find(real(etas)<0))+sqrt(etas(find(real(etas)<0)).^2-1));
rs=abs(xs).^2;
%rs(2755)*10^4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Convolution 1 (sample rocking curve  gaussian)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%
% Mono rocking curve
%%%%%%%%%%%%%%%%%%%%

bm=-1;
Pm=1.0;
ewidthm=energy*abs(real(chihm)*Pm)/sin(thbm)^2/sqrt(abs(bm));
dem1=-4*ewidthm;
dem2=10*ewidthm;
nstepm=round((dem2-dem1)/degaus);
a1=[1:nstepm];
dem=dem2-(a1-1)*degaus; % Inverted for convolution
etam=(2*bm*dem*sin(thbm)^2/energy-chi0m*(1-bm)/2)/Pm/sqrt(abs(bm)*chihm*chihbm);
xm=0*ones(nstepm,1);
xm(find(real(etam)>=0))=sqrt(abs(bm))*(sqrt(chihm*chihbm)/chihbm)*(etam(find(real(etam)>=0))-sqrt(etam(find(real(etam)>=0)).^2-1));
xm(find(real(etam)<0))=sqrt(abs(bm))*(sqrt(chihm*chihbm)/chihbm)*(etam(find(real(etam)<0))+sqrt(etam(find(real(etam)<0)).^2-1));
rm=abs(xm).^2;

rmn=(rm.*rm)/sum(rm.*rm);
cfwhmm=0.5*(dem(max(find(rm>=(max(rm)/2))))+dem(min(find(rm>=(max(rm)/2)))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convolution 2 (mono rocking curve * (sample rocking curve * gaussian))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



rsgm=conv(rmn,rs);
%rsgm(3500)*10^4
%pause
%%%%%%%%%%%%%%%%%%%
% Fit rocking curve
%%%%%%%%%%%%%%%%%%%

% X0=datar(:,1);
% Y=datar(:,2);


% Estimate initial fitting parameters for rocking curve

escale=width;
eoffset=energy;%X0(find(Y==max(Y)))-0.6;
rscale=1;%(max(datar(:,2))-0.5*(datar(size(datar,1),2)+datar(1,2)))/0.9*3;
rbgoffset=0;%0.5*(datar(size(datar,1),2)+datar(1,2))/rscale;
rbgslope=0;%(datar(end,2)-datar(1,2))/(datar(end,1)-datar(1,1))/rscale;
steps = 100;
 
%for nn = 1:100
    %factor = nn/10;
    multiples_p = [10; 1/1e3; 1/1e4; 1e3; 1e2];
p0 = [escale*multiples_p(1);eoffset*multiples_p(2);rscale*multiples_p(3);0*multiples_p(4);rbgoffset*multiples_p(5)];
%pause
lb = [0*multiples_p(1);(eoffset-1)*multiples_p(2);0*multiples_p(3);-1.0;0*multiples_p(5)];
ub = [0.3*multiples_p(1);(eoffset+1)*multiples_p(2);rscale*10*multiples_p(3);1.0;rbgoffset*10*multiples_p(5)];

option = optimoptions('fsolve','Algorithm','levenberg-marquardt');

%[p,v]=lsqnonlin(@(p) f1(p),p0,lb,ub);
p = p0;
[v] = f1(p0);
p=[p(1)/multiples_p(1); p(2)/multiples_p(2); p(3)/multiples_p(3); p(4)/multiples_p(4); p(5)/multiples_p(5)];
%p = [0.9998743 -2975.4438 15596.844 -0.0022689 0.000315];
%v = 0.2;
% datarf=datar;
% datarf(:,1)=datar(:,1)-p(2);
% datarf(:,2)=datar(:,2)/p(3)-p(4)*datarf(:,1)-p(5);
%datarf(:,3)=datarf(:,2).*datar(:,3)./datar(:,2);
dedth=energy*(pi/180)*cos(thbs)/sin(thbs);


% Screen output of fitting results for rocking curve
if plotter > 0
    disp(['                      dE/dth = ' num2str(dedth) ' eV/deg']);
    disp(['                Least-square = ' num2str((v))]);
    disp(['fitted Gaussian width = p(1) = ' num2str(p(1))]);
    disp(['        Energy offset = p(2) = ' num2str(p(2))]);
    disp(['   Incident intensity = p(3) = ' num2str(p(3))]);
    disp(['   R background slope = p(4) = ' num2str(p(4))]);
    disp(['  R background offset = p(5) = ' num2str(p(5))]);
    disp(['                        Rmax = ' num2str(max(rsgm))]);
   
    
    titleline=['Energy = ' num2str(energy) ' eV; mono = Si(' num2str(hm(1)) ' ' num2str(hm(2)) ' ' num2str(hm(3))+'); sample = (' num2str(hs(1)) ' ' num2str(hs(2)) ' ' num2str(hs(3)) '); dE/dth = ' num2str(dedth) ' (ev/deg)'];
    rline1=[];
    rline2=[];
    rline1=[' bs = ' num2str(bs) '; bm = ' num2str(bm) '; Gaussian width = ' num2str(fwhmgaus) ' (eV); Gaussian area = ' num2str(areagaus)]; 
    rline2=['Rmax = ' num2str(max(rsgm)) ';   I0 = ' num2str(p(3)) '(cts)'];
end

%%%%%%%%%%%%%%%%%%%%%
% Fit XSW yield curve
%%%%%%%%%%%%%%%%%%%%%

% datayf=datay;
% datayf(:,1)=datay(:,1)-p(2);
noffbragg=5;
% datayf(:,2)=datay(:,2)*2*noffbragg/sum(datay([1:noffbragg,size(datay,1)-noffbragg+1:size(datay,1)],2));
% X=datayf(:,1);
% Y=datayf(:,2);


% lb = [0;0;0];
% ub = [4;2;2];
% step = 50;
% for nn = 1:2*step
%     for ll = 1:step
%         fh = (nn-1)/step;
%         ph = (ll-1)/step;
%         res(nn,ll)=sum(abs(f2([1,fh,ph])));
%     end
% end

q0 = [1;fh0;ph0];

v2 = f2(q0);

q=[1;fh0;ph0];

ys=rs*C1+2*C2*q(2)*sqrt(rs).*cos(C3+atan2(imag(xs),real(xs))-2*pi*q(3));
gaus=exp(-4*log(2)*(egaus/fwhmgaus).^2);
gaus=gaus/sum(gaus);
gaus=gaus*areagaus;
if fwhmgaus>0 
  ysg=conv(gaus,ys);
else
  ysg=ys;
end
ysgm=conv(rmn,ysg)+1;

% Screen output of fitting results for yield
if plotter > 0
    display(['               Least-square = ' num2str((v2))]);
    display(['                  fH = q(2) = ' num2str(q(2))]);
    display(['                  PH = q(3) = ' num2str(q(3))]);
    display(['    Non-dipolar corrections used:'])
    display(['           C1 = (1+Q)/(1-Q) = ' num2str(C1)]);
    display(['           C2 =   1/(1-Q)   = ' num2str(C2)]);
    display(['           C3 = phase shift = ' num2str(C3)]);
end


%clf();
%fig=figure;
%fig.auto_resize='off';
%fig.axes_size=[1200,600];
%subplot(1,2,1);
%ax=gca();
%ax.margins=[0.15,0.0,0.1,0.15];
%plot2d(datarf(:,1),datarf(:,2),axesflag=2,style=-9);
%plot2d(datayf(:,1),datayf(:,2),style=-9);
%plot2d(desgm,rsgm,rect=[min(datarf(:,1)),-0.05,max(datarf(:,1)),max(datayf(:,2))*1.1]);
%plot2d(desgm,ysgm,rect=[min(datarf(:,1)),-0.05,max(datarf(:,1)),max(datayf(:,2))*1.1]);
%ax.font_style=6;
%ax.font_size=3;
%ax.x_label.text='E - E_Bragg (eV)';
%ax.x_label.font_size=4;


%%%subplot(1,2,2);
%%%ax=gca();
%%%ax.margins=[0.01,0.0,0.1,0.15];

%plot2d([0,1],[0,1],style=0,frameflag=2,axesflag=0);
%datafile2=strsubst(datafile,'\','/');

%xtex=0.02;ytex=0.05;atex=0.1;
%ax.font_size=2;
%xstring(xtex,ytex+7*atex,datafile2);
%xstring(xtex,ytex+6*atex,titleline);
%xstring(xtex,ytex+5*atex,rline1);
%xstring(xtex+0.04,ytex+4*atex,rline2);
%xstring(xtex+0.04,ytex+3*atex,rline3);
%xstring(xtex+0.01,ytex+2*atex,yline1);
%xstring(xtex+0.01,ytex+1*atex,yline2);
%xstring(xtex+0.04,ytex+0*atex,yline3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data, fit and figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
desg=des(1)-(rangegaus+egaus(1))+a1*degaus;
desgm=desg(1)-(max(dem)-cfwhmm)+a1*degaus;
% ifit=find((desgm>=min(datarf(:,1)))&(desgm<=max(datarf(:,1))))';
% ifit=[(min(ifit)-1);ifit;(max(ifit)+1)];
% etadata=(2*bs*datarf(:,1)*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
% etafit=(2*bs*desgm(ifit)'*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
% e2th=1/(energy*1e-6*cos(thbs)/sin(thbs));
% dataout=[real(etadata),datarf,datayf(:,2)];
% %fitout=[real(etafit),desgm(ifit)',rsgm(ifit)',ysgm(ifit)'];
% datatitle='eta E-Eb(eV) R_data Y_data';
% fittitle='eta E-Eb(eV) R_fit Y_fit';
%fprintfMat(dataoutfile,dataout,'%.6f',datatitle);
%fprintfMat(fitoutfile,fitout,'%.6f',fittitle);
%xs2pdf(0,figurefile);


    
%%%%%%%%%%%%%%%%%%    
%Applied Functions
%%%%%%%%%%%%%%%%%%
    
function e=f1(p) % Function defining the difference between the theoretical and measured rocking curves (Chi^2 can be potentially introduced here)
    p=[p(1)/multiples_p(1); p(2)/multiples_p(2); p(3)/multiples_p(3); p(4)/multiples_p(4); p(5)/multiples_p(5)];
        %p0 = [escale*10;eoffset/1e3;rscale/1e4;rbgslope*1e5;rbgoffset*1e4];
    fwhmgaus = p(1);
    %if fwhmgaus > (max(X0)-min(X0))/6
    %    fwhmgaus = (max(X0)-min(X0))/6;
    %end
    if fwhmgaus > 0
        gaus=exp(-4*log(2)*(egaus/fwhmgaus).^2);
        gaus=gaus/sum(gaus);
        gaus=gaus*areagaus;
        a1=[1:(nsteps+ngaus-1)];
        desg=des(1)-(rangegaus+egaus(1))+a1*degaus;
        rsga=conv(gaus,rsgm);
        %figure(3)
        %rsg(900)
    else
        desg=des;
        rsga=rsgm;
    end   
    a1=[1:(nstepm+size(desg,2)-1)];
    desgm=desg(1)-(max(dem)-cfwhmm)+a1*degaus;

    
        
    e=desgm;


end

function e=f2(q) % Function defining the difference between the theoretical and measured yield curves (Chi^2 can be potentially introduced here)
   % whos xs
   %xs
%rs*C1
  ys=rs*C1+2*C2*q(2)*sqrt(rs).*cos(C3+atan2(imag(xs),real(xs))-2*pi*q(3));
  
  if fwhmgaus>0 
    ysg=conv(gaus,ys); % Convolution of yield with a Gaussian function
  else
    ysg=ys;
  end

  ysgm=conv(rmn,ysg)+1; % Convolution of the Gaussian convoluted yield with mono rocking curve
  
  %fwhmgaus
  %figure(5)
  %hold off
  %plot(gaus)
  %hold on
  %plot(ys)
  %hold off
  
  %figure(6)
  %plot(ysg)
  %figure(7)
  %plot(ysgm)
  %figure(8)
  %hold off
  %plot(X,Y)
  %hold on
  %plot(desgm,ysgm)
  %pause(0.1)  
  
  e=ysgm;

end

rsga=conv(gaus,rsgm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   Alternative Plotter   %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotter == 1
    figure(100)
    close(100)
    figure(100)


          %make a nice picture (only on my laptop)
            %set(gca,'BackgroundColor',[1 1 1])

          %limit the plot range on x-axis, auto on y axis

  

          %plot XSW profiles (relative to bragg reflection)

            
          %plot fits (relative to bragg reflection)
          hold off
            plot(desgm,rsga, 'color', 'k', 'Linewidth', 3)
            hold on
            plot(desgm,ysgm, 'color', 'r', 'Linewidth', 3)
            hold off

            

          %label plots
            set(gca,'fontsize',20)
            xlabel('E - E_{Bragg} (eV)','FontSize',20,'FontWeight','bold','Color','k')
            ylabel('Relative Absorption','FontSize',20,'FontWeight','bold','Color','k')

%             text(0.6,1,[...
%                    '                 Filename: ' dataprefix char(10) char(10)...
%                    'General Information: ' char(10)...
%                    '   Energy = ' num2str(energy) ' eV,      mono = Si(' num2str(hm(1)) ' ' num2str(hm(2)) ' ' num2str(hm(3)) '),' char(10)... 
%                    '   sample = ' samp '(' num2str(hs(1)) ' ' num2str(hs(2)) ' ' num2str(hs(3)) '),     dE/dth = ' num2str(dedth) ' (ev/deg)' char(10)...
%                    'Fitting the Bulk Reflection Profile: ' char(10)...
%                    '   bs = ' num2str(bs) ',       bm = ' num2str(bm) char(10)...
%                    '   Gaussian width = ' num2str(fwhmgaus) ' (eV),   Gaussian area = ' num2str(areagaus) char(10)...
%                    '   Rmax = ' num2str(max(rsgm)) ',                 I0 = ' num2str(p(3)) '(cts)' char(10)...
%                    'Fitting the XSW Absorption Curve: ' char(10)...
%                    '   fH = ' num2str(q(2))  char(10)...
%                    '   PH = ' num2str(q(3))  char(10)...
%                    'Non-dipolar corrections used: ' char(10)...
%                    '   C1 = (1+Q)/(1-Q) = ' num2str(C1) char(10)...
%                    '   C2 = 1/(1-Q) = ' num2str(C2) char(10)...
%                    '   C3 = phase shift = ' num2str(C3) char(10)... 
%                    '   Q =  ' num2str(Q) char(10)], 'interpreter', 'none') % 'interpreter', 'none 'allows to do the underscore
            hold off
            axis([-3 5 -0.1 3.5])
%             savefig([dataprefix '.fig'])
%             saveas(gcf,dataprefix,'pdf')
%             saveas(gcf,dataprefix,'svg')
end

the_out = [desgm',ysgm'];

the_nix_out = [desgm(:),rsga(:)];

fit_out = [{the_out},{the_nix_out}];
q(1) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   Alternative Plotter   %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   write out data        %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[filename,pathname] = uiputfile('*.txt', 'Save Fit Data As');

%coherent fraction
%dlmwrite([pathname filename], ' ', 'delimiter', '', 'newline', 'pc', '-append')
%dlmwrite([pathname filename], 'CoherentFraction', 'delimiter', '', 'newline', 'pc', '-append', 'precision', 10)
%dlmwrite([pathname filename], q(2), 'newline', 'pc', '-append')

%coherent position
%dlmwrite([pathname filename], ' ', 'delimiter', '', 'newline', 'pc', '-append')
%dlmwrite([pathname filename], 'CoherentPosition', 'delimiter', '', 'newline', 'pc', '-append', 'precision', 10)
%dlmwrite([pathname filename], q(3), 'newline', 'pc', '-append')

%RelativePhotonEnergy XSWyield    XSWreflection
%dlmwrite([pathname filename], ' ', 'delimiter', '', 'newline', 'pc', '-append')
%dlmwrite([pathname filename], 'RelativePhotonEnergy XSWyield    XSWreflection', 'delimiter', '', 'newline', 'pc', '-append', 'precision', 10)
%clear COMBINE;
%COMBINE = [datarf(:,1), datayf(:,2), datarf(:,2)];
%dlmwrite([pathname filename], COMBINE, '-append', 'delimiter', '\t', 'newline', 'pc','precision', 10)

%RelativePhotonEnergy XSWyieldFit
%dlmwrite([pathname filename], ' ', 'delimiter', '', 'newline', 'pc', '-append')
%dlmwrite([pathname filename], 'RelativePhotonEnergy    XSWyieldFIT', 'delimiter', '', 'newline', 'pc', '-append', 'precision', 10)
%clear COMBINE;
%COMBINE = [desgm',ysgm];
%dlmwrite([pathname filename], COMBINE, '-append', 'delimiter', '\t', 'newline', 'pc','precision', 10)

%RelativePhotonEnergy XSWreflectionFit
%dlmwrite([pathname filename], ' ', 'delimiter', '', 'newline', 'pc', '-append')
%dlmwrite([pathname filename], 'RelativePhotonEnergy    XSWreflectionFIT', 'delimiter', '', 'newline', 'pc', '-append', 'precision', 10)
%clear COMBINE;
%COMBINE = [desgm',rsgm];
%dlmwrite([pathname filename], COMBINE, '-append', 'delimiter', '\t', 'newline', 'pc','precision', 10)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   write out data        %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
