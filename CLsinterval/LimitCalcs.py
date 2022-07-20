from POIEstimator import POIEstimator
import ROOT

sigdata_1TeV = [0.463,324.7,469.0,405.2,176.9,12.09,1.21,0.226,0.169,0.0678,0.0113]
sigdata_2TeV = [0.007,5.036,8.394,10.02,10.62,10.52,9.395,7.704,5.211,1.934,0.289]
sigdata_3TeV = [0.004,0.4123,0.5734,0.6702,0.72,0.7553,0.764,0.769,0.732,0.678,0.598]
sigdata_4TeV = [0.00005,0.026,0.023,0.017,0.014,0.012,0.010,0.11,0.009,0.0087,0.008]

bkgdata  = [6.628,930.51,310.02,34.17,5.75,3.55,1.789,0.715,0.279,0.039,0.0398]
bkg_uncerts=[1.105, 12.86,7.10,2.19,0.576,0.433,0.267,0.169,0.105,0.0397,0.0397]

est_M1TeV = POIEstimator(sigdata_1TeV,bkgdata,bkg_uncerts)
est_M2TeV = POIEstimator(sigdata_2TeV,bkgdata,bkg_uncerts)
est_M3TeV = POIEstimator(sigdata_3TeV,bkgdata,bkg_uncerts)
est_M4TeV = POIEstimator(sigdata_4TeV,bkgdata,bkg_uncerts)

poi_val1 = est_M1TeV.POIrange(0.01,1,50)
poi_val2 = est_M2TeV.POIrange(0.01,1,50)
poi_val3 = est_M3TeV.POIrange(0.01,1,50)
poi_val4 = est_M4TeV.POIrange(0.00001,1,50)

exp_limits_1 = est_M1TeV.compute_limits(poi_val1,0.05)
exp_limits_2 = est_M2TeV.compute_limits(poi_val2,0.05)
exp_limits_3 = est_M3TeV.compute_limits(poi_val3,0.05)
exp_limits_4 = est_M4TeV.compute_limits(poi_val4,0.05)

limits = [exp_limits_1,exp_limits_2,exp_limits_3]#,exp_limits_4]
xsec = [2.457*1000,0.107*1000,0.011*1000,0.00162*1000]

#print(exp_limits_4)

labels = [1000,2000,3000]#,4000]
N = len(labels) 
yellow = ROOT.TGraph(2*N)
green  = ROOT.TGraph(2*N)
median = ROOT.TGraph(N)

up2s = [ ]
for i in range(N):
    up2s.append(limits[i][4]*xsec[i])
    print(i,'  ',limits[i][2]*xsec[i])
    yellow.SetPoint(i,  labels[i] , limits[i][4]*xsec[i])
    green.SetPoint( i,  labels[i] , limits[i][3]*xsec[i])
    median.SetPoint(i,  labels[i] , limits[i][2]*xsec[i])
    green.SetPoint(  2*N-1-i, labels[i], limits[i][1]*xsec[i]) 
    yellow.SetPoint( 2*N-1-i, labels[i], limits[i][0]*xsec[i])

W = 800
H  = 600
T = 0.08*H
B = 0.12*H
L = 0.12*W
R = 0.04*W
c = ROOT.TCanvas("c","c",100,100,W,H)

c.SetFillColor(0)
c.SetBorderMode(0)
c.SetFrameFillStyle(0)
c.SetFrameBorderMode(0)
c.SetLeftMargin( L/W )
c.SetRightMargin( R/W )
c.SetTopMargin( T/H )
c.SetBottomMargin( B/H )
c.SetTickx(0)
c.SetTicky(0)
#c.SetGrid()
c.SetLogy()
c.cd()
frame = c.DrawFrame(1.4,0.001, 4.1, 10)
frame.GetYaxis().CenterTitle()
frame.GetYaxis().SetTitleSize(0.05)
frame.GetXaxis().SetTitleSize(0.05)
frame.GetXaxis().SetLabelSize(0.04)
frame.GetYaxis().SetLabelSize(0.04)
frame.GetYaxis().SetTitleOffset(0.9)
frame.GetXaxis().SetNdivisions(508)
frame.GetYaxis().CenterTitle(True)
frame.GetYaxis().SetTitle("95% upper limit on #sigma #times B(W'#rightarrow #tau#nu) [fb] ")
#    frame.GetYaxis().SetTitle("95% upper limit on #sigma #times BR / (#sigma #times BR)_{SM}")
frame.GetXaxis().SetTitle("Mw' [GeV]]")
frame.SetMinimum(1)
frame.SetMaximum(max(up2s)*1.05)
frame.GetXaxis().SetLimits(labels[0],labels[-1]+100)

yellow.SetFillColor(ROOT.kOrange)
yellow.SetLineColor(ROOT.kOrange)
yellow.SetFillStyle(1001)
yellow.Draw('F')
 
green.SetFillColor(ROOT.kGreen+1)
green.SetLineColor(ROOT.kGreen+1)
green.SetFillStyle(1001)
green.Draw('Fsame')
 
median.SetLineColor(1)
median.SetLineWidth(2)
median.SetLineStyle(2)
median.Draw('lsame')

x1 = 0.5
x2 = x1 + 0.24
y2 = 0.90
y1 = 0.75
legend = ROOT.TLegend(x1,y1,x2,y2)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.041)
legend.SetTextFont(42)
legend.AddEntry(median, "Asymptotic CL_{s} expected",'L')
legend.AddEntry(green, "#pm 1 std. deviation",'f')
#    legend.AddEntry(green, "Asymptotic CL_{s} #pm 1 std. deviation",'f')
legend.AddEntry(yellow,"#pm 2 std. deviation",'f')
#    legend.AddEntry(green, "Asymptotic CL_{s} #pm 2 std. deviation",'f')
legend.Draw()
 
print(" ")
c.SaveAs("UpperLimit.png")
c.Close()
