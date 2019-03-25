This is a poster template for an A0 poster with Durham Uni style branding.  If you want a smaller poster it is best to make it A0 and then shrink it when you print.

The method to make the everything work is a bit involved.  All of the logos are .eps, and any pictures you want to include also have to be .eps.  The University .jpg files don't like being blown up to A0 size, and they look pixelated when printed.

To create a .pdf file for printing:
1) LaTeX the file to .dvi.
2) Use the converter in e.g. Kate to change this to a .ps.
3) Use the terminal to convert to a pdf e.g.
	ps2pdf myfile.ps myfile.pdf
or
	ps2pdf myfile.ps
if you want the two files to have the same name.

You now have a beautiful pdf poster.

If you make any good changes to the template, or know how to make compilation simpler, let me know.

John
john.chapman@dur.ac.uk
