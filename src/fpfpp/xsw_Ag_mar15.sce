// Tien-Lin Lee, Diamond Light Source
// This Scilab script does least-square fitting of theoretical
// rocking curves and XSW yield curves to data in energy. The 
// calculations are based on dynamical diffraction theory with the 
// possibility of convoluting the sample rocking curve with a
// Si monochromator rocking curve and a Gaussian function
// accounting for sample imperfection. Non-dipolar corrections 
// can be also included.

script_dir="C:\Users\szx13119\Dropbox\Data\FePc_i09\sci_lab_code\"; // Folder containing (1) nff file(s) for substrate, (2) 'si.nff', (3) 'f0_all_free_atoms.txt' and (4) 'trzhaskovskaya_at_data_nucl_data_tab.txt'
dataprefix="test"; // File prefix for reflectivity data (usually named by script 'nxs3D2txt'); Extension should be '.txt';
datadir="C:\Users\szx13119\Dropbox\Data\FePc_i09\sci_lab_code\"+dataprefix+"-txt\"; // Folder containing reflectivity and xps data created by script 'nxs3D2txt'; replace this with the full path if the folder is named differently;
//xpsfile="Co2p.TXT"; // Multi-column XPS data file; can be created by CasaXPS; photon energy column is not needed but the energy steps must match those in the reflectivity data file.
xpscol=7; // Column number in the XPS data file that contains the core-level data to be analysed
xpselem='N'; // String, element to be analysed; case sensitive
xpscorelevel='1s'; // String, core-level of the element to be analysed; case sensitive and no space
xpslabel="779.5eV_test"; // Additional label, e.g., binding energy, to be used to identify a chemical component

samp="Ag"; // Chemical formula of sample crystal; this does not affect the XSW calculations
hs=[1,1,1]; // Sample reflection
energy=2628; // Photon energy in eV
fwhmgaus=0.3; // in eV, FWHM of the Gaussian to be convoluted with the sample theoretical rocking curve (to account1 for, e.g., mosaicity).
fh0=0.1; // Initial value of coherent fraction
ph0=0.9; // Initial value of coherent position
NDC=1; // Non-dipolar corrections; On: NDC = 1; Off: NDC = 0.

hm=[1,1,1]; // Reflection of Si monochromator
N_monobounce=2; // Number of bounces of Si monochromator reflection; 2 for DCM and 4 for 4-bounce mono

z=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No'];

// Non-dipolar corrections, assuming delta=0
// C1=(1+Q)/(1-Q), C2=1/(1-Q), C3=phase shift=0
// Ref[1]: Trzhaskovskaya et al., Atomic Data and Nuclear Data Tables 77, 97 (2001)
// Analyser-beam geometry:
thdeg=15 // Theta angle in FIG1 in Ref[1]
phideg=0; // Phi angle in FIG1 in Ref[1]
the=thdeg*%pi/180;
phie=phideg*%pi/180;

if find(z==xpselem)==[] then
    mprintf('\nElement %s does not exist!\n',xpselem);
    abort;
else
    ndcfile=mopen(script_dir+"trzhaskovskaya_at_data_nucl_data_tab.txt",'rt');
    strfound=0;
    sbgd9=zeros(5,9);
    while strfound==0,
        header=mgetl(ndcfile,1);    
        if grep(header,string(find(z==xpselem))+' '+xpscorelevel)==1 then
            Eb=mfscanf(ndcfile,'%d');
            for j=1:5
                for i=1:9
                    sbgd9(j,i)=mfscanf(ndcfile,'%e');end;end;
            strfound=1;
        elseif header==[] then
            mprintf('\nCore-level %s %s does not exist in database!\n',xpselem,xpscorelevel);
            abort;
        else 
            mgetl(ndcfile,6);end;
    end;
    mclose(ndcfile);

    bet=interpln(sbgd9([1,3],:),energy-Eb);
    gam=interpln(sbgd9([1,4],:),energy-Eb);
    Q=gam*cos(the)^2*sin(the)*cos(phie)/(1+bet*0.5*(3*cos(the)^2-1));
    C1=(1+Q)/(1-Q);C2=1/(1-Q);C3=0;end;

if NDC==0 then
    C1=1;C2=1;C3=0;end;

rfile=datadir+dataprefix+"-r.txt";
yfile=datadir+xpsfile;
//r0=fscanfMat(rfile);
//xps0=fscanfMat(yfile);
dataoutfile=datadir+dataprefix+"_"+xpselem+strsubst(xpscorelevel,'/','')+"_"+xpslabel+"_dat"+".txt";
fitoutfile=datadir+dataprefix+"_"+xpselem+strsubst(xpscorelevel,'/','')+"_"+xpslabel+"_fit"+".txt";
figurefile=datadir+dataprefix+"_"+xpselem+strsubst(xpscorelevel,'/','')+"_"+xpslabel+".pdf";

lambda=12398.54/energy; // Wavelength in A

// Read in data files
data=data_hold;//[r0(:,1),xps0(:,xpscol),r0(:,2)];
dataE=gsort(data(:,1),'g','i');
Esi=[];
for i=1:size(dataE,1)
    Esi=[Esi,find(data(:,1)==dataE(i,1))];
end;
datar=data(Esi,[1,3]);
xshift=0;
datay=data(Esi,[1:2]);
datay(:,1)=datay(:,1)+xshift;

// Lattice parameters, DW factor and atomic coordinates, sample
lps0=[4.08516,4.08516,4.08516,90,90,90]; // a,b and c in A; alpha, beta and gamma in deg, sample.
DWBs=0.; // Debye-Waller B factor, sample.
xyzs=[
47,0,0,0,1; // Atomic number,x,y,z,occupancy 
47,1/2,1/2,0,1;
47,1/2,0,1/2,1;
47,0,1/2,1/2,1;];
nas=size(xyzs,1);

// Lattice parameters, DW factor and atomic coordinates, Si monochromator
lpm0=[5.431,5.431,5.431,90,90,90]; // a,b and c in A; alpha, beta and gamma in deg, mono.
DWBm=0.; // Debye-Waller B factor, mono.
xyzm=[
14,0,0,0,1; // Atomic number,x,y,z,occupancy
14,0.5,0.5,0,1;
14,0.5,0,0.5,1;
14,0,0.5,0.5,1;
14,0.25,0.25,0.25,1;
14,0.75,0.75,0.25,1;
14,0.75,0.25,0.75,1;
14,0.25,0.75,0.75,1;
];// Mono: Si
nam=size(xyzm,1);
xyzm(:,2:4)=xyzm(:,2:4)-ones(nam,1)*[0.125,0.125,0.125]; // Shift the origin to inversion center 

// Unit cell volumes, Bragg plane spacings and Bragg angles
lps=lps0;lps(1,4:6)=lps(1,4:6)*%pi/180; // Change deg to rad, sample.
lpm=lpm0;lpm(1,4:6)=lpm(1,4:6)*%pi/180; // Change deg to rad, mono.
ucvs=lps(1)*lps(2)*lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6))); // Unit cell volume in A^3, sample.
ucvm=lpm(1)*lpm(2)*lpm(3); // Unit cell volume in A^3, mono Si.
lvs=[lps(1),0,0;
lps(2)*cos(lps(6)),lps(2)*sin(lps(6)),0;
lps(3)*cos(lps(5)),lps(3)*(cos(lps(4))-cos(lps(5))*cos(lps(6)))/sin(lps(6)),lps(3)*sqrt(1-cos(lps(4))^2-cos(lps(5))^2-cos(lps(6))^2+2*cos(lps(4))*cos(lps(5))*cos(lps(6)))/sin(lps(6))];// Real space lattice vectors a, b, and c in Cartesian coordinates with a parallel to X and b in the XY plane

rlvs=[lvs(2,2)*lvs(3,3)-lvs(2,3)*lvs(3,2),lvs(2,3)*lvs(3,1)-lvs(2,1)*lvs(3,3),lvs(2,1)*lvs(3,2)-lvs(2,2)*lvs(3,1);
lvs(3,2)*lvs(1,3)-lvs(3,3)*lvs(1,2),lvs(3,3)*lvs(1,1)-lvs(3,1)*lvs(1,3),lvs(3,1)*lvs(1,2)-lvs(3,2)*lvs(1,1);
0,0,lps(1)*lps(2)*sin(lps(6))]/ucvs;// Reciprocal space lattice vectors a*, b*, and c* in Cartesian coordinates

dhs=sqrt(sum((hs*rlvs).^2)).^(-1);// Bragg plane spacing for hkl reflection in A, sample.
dhm=lpm(1)/sqrt(hm*hm');// Bragg plane spacing for hkl reflection in A, mono Si.

thbs=asin(dhs^(-1)*lambda/2); // Bragg angle in rad, sample.
thbm=asin(dhm^(-1)*lambda/2); // Bragg angle in rad, mono Si.

// Structure factors and chi values, sample
f0=fscanfMat(script_dir+"f0_all_free_atoms.txt");
fps=ones(nas,1)*0;fpps=ones(nas,1)*0;
fs=ones(nas,1)*0;f0s=ones(nas,1)*0;
for i=1:nas,
  fpfppdata=fscanfMat(script_dir+string(z(xyzs(i,1)))+'.nff');
  fps(i)=interpln([fpfppdata(:,1),fpfppdata(:,2)]',energy)-xyzs(i,1);
  fpps(i)=interpln([fpfppdata(:,1),fpfppdata(:,3)]',energy);
  fs(i)=interpln([f0(:,1),f0(:,xyzs(i,1)-1)]',0.5*dhs^(-1))+fps(i)+%i*fpps(i);
  f0s(i)=xyzs(i,1)+fps(i)+%i*fpps(i);
end;

hrs=xyzs(:,2:4)*hs';
Fhs=(exp(2*%pi*%i*hrs')*(fs.*xyzs(:,5)))*exp(-DWBs/dhs^2/4);
Fhbs=(exp(-2*%pi*%i*hrs')*(fs.*xyzs(:,5)))*exp(-DWBs/dhs^2/4);
F0s=sum(f0s.*xyzs(:,5))*exp(-DWBs/dhs^2/4);
	
gams=2.818e-5*lambda^2/%pi/ucvs;
chihs=-gams*Fhs;
chihbs=-gams*Fhbs;
chi0s=-gams*F0s;

// Structure factors and chi values, mono
fpm=ones(nam,1)*0;fppm=ones(nam,1)*0;
fm=ones(nam,1)*0;f0m=ones(nam,1)*0;
for i=1:nam,
  fpfppdata=fscanfMat(script_dir+string(z(xyzm(i,1)))+'.nff');
  fpm(i)=interpln([fpfppdata(:,1),fpfppdata(:,2)]',energy)-xyzm(i,1);
  fppm(i)=interpln([fpfppdata(:,1),fpfppdata(:,3)]',energy);
  fm(i)=interpln([f0(:,1),f0(:,xyzm(i,1)-1)]',0.5*dhm^(-1))+fpm(i)+%i*fppm(i);
  f0m(i)=xyzm(i,1)+fpm(i)+%i*fppm(i);
end;

hrm=xyzm(:,2:4)*hm';
Fhm=(exp(2*%pi*%i*hrm')*(fm.*xyzm(:,5)))*exp(-DWBm/dhm^2/4);
Fhbm=(exp(-2*%pi*%i*hrm')*(fm.*xyzm(:,5)))*exp(-DWBm/dhm^2/4);
F0m=sum(f0m.*xyzm(:,5))*exp(-DWBm/dhm^2/4);
	
gamm=2.818e-5*lambda^2/%pi/ucvm;
chihm=-gamm*Fhm;
chihbm=-gamm*Fhbm;
chi0m=-gamm*F0m;

// Gaussian (normalized to integrated area)
areagaus=1;
ngaus=101;
if fwhmgaus>0 then
  rangegaus=4.0*fwhmgaus;
  degaus=rangegaus/(ngaus-1);
  egaus=[-round((ngaus-1)/2):(ngaus-round((ngaus-1)/2)-1)]*degaus;
  gaus=exp(-4*log(2)*(egaus/fwhmgaus).^2);
  gaus=gaus/sum(gaus);
  gaus=gaus*areagaus;
else
  degaus=0.005;
end;

// Sample rocking curve
bs=-1.;
Ps=1.0;
ewidths=energy*abs(real(chihs)*Ps)/sin(abs(thbs))^2/sqrt(abs(bs));
des1=-7*ewidths;
des2=7*ewidths;
nsteps=round((des2-des1)/degaus);
a1=[1:nsteps];
des=des1+(a1-1)*degaus;
etas=(2*bs*des*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
xs=0*ones(nsteps);
xs(find(real(etas)>=0))=sqrt(abs(bs))*(sqrt(chihs*chihbs)/chihbs)*(etas(find(real(etas)>=0))-sqrt(etas(find(real(etas)>=0)).^2-1));
xs(find(real(etas)<0))=sqrt(abs(bs))*(sqrt(chihs*chihbs)/chihbs)*(etas(find(real(etas)<0))+sqrt(etas(find(real(etas)<0)).^2-1));
rs=abs(xs).^2;


// Convolution 1 (sample rocking curve * gaussian)
if fwhmgaus>0 then
  a1=[1:(nsteps+ngaus-1)];
  desg=des(1)-(rangegaus+egaus(1))+a1*degaus;
  rsg=convol(gaus,rs);
else
  desg=des;
  rsg=rs;
end;

// Mono rocking curve
bm=-1;
Pm=1.0;
ewidthm=energy*abs(real(chihm)*Pm)/sin(thbm)^2/sqrt(abs(bm));
dem1=-4*ewidthm;
dem2=10*ewidthm;
nstepm=round((dem2-dem1)/degaus);
a1=[1:nstepm];
dem=dem2-(a1-1)*degaus; // Inverted for convolution
etam=(2*bm*dem*sin(thbm)^2/energy-chi0m*(1-bm)/2)/Pm/sqrt(abs(bm)*chihm*chihbm);
xm=0*ones(nstepm);
xm(find(real(etam)>=0))=sqrt(abs(bm))*(sqrt(chihm*chihbm)/chihbm)*(etam(find(real(etam)>=0))-sqrt(etam(find(real(etam)>=0)).^2-1));
xm(find(real(etam)<0))=sqrt(abs(bm))*(sqrt(chihm*chihbm)/chihbm)*(etam(find(real(etam)<0))+sqrt(etam(find(real(etam)<0)).^2-1));
rm=abs(xm).^2;
rmNn=(rm.^N_monobounce)/sum(rm.^N_monobounce);
cfwhmm=0.5*(dem(max(find(rm>=(max(rm)/2))))+dem(min(find(rm>=(max(rm)/2)))));

// Convolution 2 (mono rocking curve * (sample rocking curve * gaussian))
a1=[1:(nstepm+size(desg,2)-1)];
desgm=desg(1)-(max(dem)-cfwhmm)+a1*degaus;
rsgm=convol(rmNn,rsg);


// Fit rocking curve
X0=datar(:,1);
Y=datar(:,2);
function e=f1(p,m) // Function defining the difference between the theoretical and measured rocking curves (Chi^2 can be potentially introduced here)
  X=p(1)*X0+p(2)
  e=Y/p(3)-p(4)*X-p(5)-interpln([desgm;rsgm],X)';
endfunction;

// Estimate initial fitting parameters for rocking curve
rscale=(max(datar(:,2))-0.5*(datar(size(datar,1),2)+datar(1,2)));
rbgoffset=0.5*(datar(size(datar,1),2)+datar(1,2))/rscale;
escale=desgm(max(find(rsgm>=0.5*max(rsgm))))-desgm(min(find(rsgm>=0.5*max(rsgm))));
escale=escale/(datar(max(find(datar(:,2)>=0.5*(max(datar(:,2))+rscale*rbgoffset))),1)-datar(min(find(datar(:,2)>=0.5*(max(datar(:,2))+rscale*rbgoffset))),1));
eoffset=0.5*(desgm(max(find(rsgm>=0.5*max(rsgm))))+desgm(min(find(rsgm>=0.5*max(rsgm)))));
eoffset=eoffset-0.5*escale*(datar(max(find(datar(:,2)>=0.5*max(datar(:,2)))),1)+datar(min(find(datar(:,2)>=0.5*max(datar(:,2)))),1));
rbgslope=(datar(size(datar,1),2)-datar(1,2))/(datar(size(datar,1),1)-datar(1,1))/escale/rscale;

[p,v,info]=lsqrsolve([escale;eoffset;rscale;rbgslope;rbgoffset],f1,size(X0,1)); // Fitting iterations for rocking curve
datarf=datar;
datarf(:,1)=datar(:,1)*p(1)+p(2);
datarf(:,2)=datar(:,2)/p(3)-p(4)*datarf(:,1)-p(5);
//datarf(:,3)=datarf(:,2).*datar(:,3)./datar(:,2);
dedth=energy*(%pi/180)*cos(thbs)/sin(thbs);

// Screen output of fitting results for rocking curve
mprintf("\n                     dE/dth = %.3e eV/deg\n",dedth);
mprintf("               Least-square = %.4e\n",norm(v)^2);
mprintf(" Energy scale factor = p(1) = %.5e\n",p(1));
mprintf("       Energy offset = p(2) = %.5e\n",p(2));
mprintf("  Incident intensity = p(3) = %.5e\n",p(3));
mprintf("  R background slope = p(4) = %.5e\n",p(4));
mprintf(" R background offset = p(5) = %.5e\n",p(5));
mprintf("                       Rmax = %.4f\n",max(rsgm));
mprintf("               Energy range = %f (eV)\n",max(datarf(:,1))-min(datarf(:,1)));
mprintf("              Angular range = %f (deg)\n",(max(datarf(:,1))-min(datarf(:,1)))/dedth);

titleline="Energy = "+string(energy)+" eV; mono = Si("+string(hm(1))+" "+string(hm(2))+" "+string(hm(3))+"); sample = "+samp+"("+string(hs(1))+" "+string(hs(2))+" "+string(hs(3))+"); dE/dth = "+string(dedth)+" (ev/deg)";
rline1=[];rline2=[];
rline1="R:  Least-square = "+string(norm(v)^2)+"; bs = "+string(bs)+"; bm = "+string(bm)+"; Gaussian width = "+string(fwhmgaus)+" (eV); Gaussian area = "+string(areagaus); 
rline2="Rmax = "+string(max(rsgm))+";   I0 = "+string(p(3))+"(cts)";
rline3="Energy range = "+string(max(datarf(:,1))-min(datarf(:,1)))+" (eV)"+";   Angular range = "+string((max(datarf(:,1))-min(datarf(:,1)))/dedth)+" (deg)";

// Fit XSW yield curve
datayf=datay;
datayf(:,1)=datay(:,1)*p(1)+p(2);
noffbragg=5;
datayf(:,2)=datay(:,2)*2*noffbragg/sum(datay([1:noffbragg,size(datay,1)-noffbragg+1:size(datay,1)],2));

X=datayf(:,1);
Y=datayf(:,2);
function e=f2(q,m) // Function defining the difference between the theoretical and measured yield curves (Chi^2 can be potentially introduced here)
  ys=rs*C1+2*C2*q(2)*sqrt(rs).*cos(C3+atan(imag(xs),real(xs))-2*%pi*q(3));
  ysg=ys;
  if fwhmgaus>0 then
    ysg=convol(gaus,ys); // Convolution of yield with a Gaussian function
  end;
  ysgm=convol(rmNn,ysg)+1; // Convolution of the Gaussian convoluted yield with mono rocking curve
  e=(q(1)*Y-interpln([desgm;ysgm],X)');
endfunction;



[q,v,info]=lsqrsolve([1;fh0;ph0],f2,size(X,1)); // Fitting iterations for yield
if q(2)<0 then q(2)=-q(2);end;
if q(3)>=1.0 then q(3)=q(3)-1.0;
elseif q(3)<0.0 then q(3)=1.0-q(3);end;

//q=[1;fh0;ph0];
//q(2)=fh0;q(3)=ph0;

datayf(:,2)=datayf(:,2)*q(1);
ys=rs*C1+2*C2*q(2)*sqrt(rs).*cos(C3+atan(imag(xs),real(xs))-2*%pi*q(3));
ysg=ys;
if fwhmgaus>0 then
  ysg=convol(gaus,ys);
end;
ysgm=convol(rmNn,ysg)+1;
yob=(datay(1,2)+datay(size(datay,1),2))*0.5/q(1);

// Screen output of fitting results for yield
mprintf("\n               Least-square = %.4e\n",norm(v)^2);
mprintf("                 Yob = q(1) = %.5e\n",yob);
mprintf("                  fH = q(2) = %.5f\n",q(2));
mprintf("                  PH = q(3) = %.5f\n",q(3));
mprintf("    Non-dipolar corrections used:\n")
mprintf("           C1 = (1+Q)/(1-Q) = %.5f\n",C1);
mprintf("           C2 =   1/(1-Q)   = %.5f\n",C2);
mprintf("           C3 = phase shift = %.5f\n",C3);

yline1=[];
yline1="Y:  Least-square = "+string(norm(v)^2)+"; Yob = "+string(yob)+"; fH = "+string(q(2))+"; PH = "+string(q(3));
if NDC==1 then
    yline2="Non-dipolar corrections for "+xpselem+' '+xpscorelevel+': On';
else
    yline2="Non-dipolar corrections for "+xpselem+' '+xpscorelevel+': Off';end;
yline3="Theta = "+string(thdeg)+" deg; Phi = "+string(phideg)+" deg";
yline4="Beta = "+string(bet)+"; Gamma = "+string(gam)+"; Q = "+string(Q);
yline5=" C1 = (1+Q)/(1-Q) = "+string(C1)+"; C2 = 1/(1-Q) = "+string(C2)+"; C3 = phase shift = "+string(C3);

scf(0);clf();
fig=gcf();
fig.auto_resize='off';
fig.axes_size=[1200,600];
subplot(1,2,1);
ax=gca();
ax.margins=[0.15,0.0,0.1,0.15];
plot2d(datarf(:,1),datarf(:,2),axesflag=2,style=-9);
plot2d(datayf(:,1),datayf(:,2),style=-9);
plot2d(desgm,rsgm,rect=[min(datarf(:,1)),-0.05,max(datarf(:,1)),max(datayf(:,2))*1.1]);
plot2d(desgm,ysgm,rect=[min(datarf(:,1)),-0.05,max(datarf(:,1)),max(datayf(:,2))*1.1]);
ax.font_style=6;
ax.font_size=3;
ax.x_label.text='E - E_Bragg (eV)';
ax.x_label.font_size=4;

xcplot=0.5*(min(datarf(:,1))+max(datarf(:,1)));
xrplot=max(datarf(:,1))-min(datarf(:,1));
ycplot=0.5*(-0.05+max(datayf(:,2))*1.05);
yrplot=max(datayf(:,2))*1.05+0.05;

subplot(1,2,2);
ax=gca();
ax.margins=[0.01,0.0,0.1,0.15];
plot2d([0,1],[0,1],style=0,frameflag=2,axesflag=0);
rfile2=strsubst(rfile,'\','/');
yfile2=strsubst(yfile,'\','/');
outputfile2=dataprefix+"_"+xpselem+strsubst(xpscorelevel,'/','')+"_"+xpslabel;

xtex=0.02;ytex=0.1;atex=0.08;
ax.font_size=2;
xstring(xtex,ytex+10*atex,'Input files:');
xstring(xtex,ytex+9.5*atex,rfile2);
xstring(xtex,ytex+9*atex,yfile2);
xstring(xtex,ytex+8*atex,'Output (fit, data and pdf) files:');
xstring(xtex,ytex+7.5*atex,outputfile2);
xstring(xtex,ytex+6.5*atex,titleline);
xstring(xtex,ytex+5.5*atex,rline1);
xstring(xtex+0.04,ytex+5*atex,rline2);
xstring(xtex+0.04,ytex+4.5*atex,rline3);
xstring(xtex,ytex+3.5*atex,yline1);
xstring(xtex,ytex+2.5*atex,yline2);
xstring(xtex+0.04,ytex+2*atex,yline3);
xstring(xtex+0.04,ytex+1.5*atex,yline4);
xstring(xtex+0.04,ytex+1.0*atex,yline5);

// Save the data, fit and figure
ifit=find((desgm>=min(datarf(:,1)))&(desgm<=max(datarf(:,1))))';
ifit=[(min(ifit)-1);ifit;(max(ifit)+1)];
etadata=(2*bs*datarf(:,1)*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
etafit=(2*bs*desgm(ifit)'*sin(thbs)^2/energy-chi0s*(1-bs)/2)/Ps/sqrt(abs(bs)*chihs*chihbs);
e2th=1/(energy*1e-6*cos(thbs)/sin(thbs));
dataout=[real(etadata),datarf,datayf(:,2)];
fitout=[real(etafit),desgm(ifit)',rsgm(ifit)',ysgm(ifit)'];
datatitle="eta E-Eb(eV) R_data Y_data";
fittitle="eta E-Eb(eV) R_fit Y_fit";
fprintfMat(dataoutfile,dataout,"%.6f",datatitle);
fprintfMat(fitoutfile,fitout,"%.6f",fittitle);
xs2pdf(0,figurefile);
