import os
os.system('mkdir selected')
for i in range(1, 40):
     if i < 10:
             b = "yaleB0" + ("%d") % i
     else:
             b = "yaleB" + ("%d") % i
     print b

     try:
	     os.chdir(b);
	     print os.getcwd();
	     os.system("cp  `ls| grep " + b + "_P00A.0[012].E.[012]..pgm` ../selected/")
	     os.system("cp  `ls| grep " + b + "_P00A+035E+15.pgm` ../selected/")
	     
	     os.system("ls| grep " + b + "_P00A.0[012].E.[012]..pgm | wc -l")
	     os.chdir("..");
	     print os.getcwd();
     except:
	 	pass