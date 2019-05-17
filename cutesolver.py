
# cutesolver.py  - løser den dæmpede fjederligning numerisk.

import numpy as np
import matplotlib.pyplot as plt

def cutesolver(parametre=0,startværdier=0,Dt=0.1,tmax=100.,v_potens=1.):
    '''
    Løser en 'fjeder ligning', det vil sige en differentialligning af typen
    mx'' = -b(x')^p-kx

    Hvis p=1 afhænger dæmpningen af hastigheden i 1.potens (fx indre modstand).
    Hvis p=2 så afhænger dæmpningen af hastigheden i anden potens,
    hvilket svarer til fx luftmodstand på loddet.

    Brug:
    cutesolver()
    kører en p=1 model med standard parametre og startværdier.

    cutesolver(parametre=(2,0.25,0.5))
    kører en p=1 model med masse 2 kg, fjederkonstant 0.25 N/m, og dæmpningskoefficent 0.5.

    cutesolver(Dt=0.001,tmax=100)
    kører en p=1 model med standard parametre, op til 100 sekunder, med tidsskridt 0.001 sekund.

    cutesolver(Dt=0.01,tmax=100,v_potens=2)
    kører en p=2 model.
    '''

    # denne her del skaffer default værdier for startværdier og fjeder-parametre hvis nødvendigt.
    if parametre == 0:
        parametre = def_param()
    if startværdier == 0:
        startværdier = def_start()

    # vi definerer en 'counter', der holder styr på hvor mange tidsskridt programmet har udført.
    # den skal starte på 0, og så skal den stige med 1 for hver tidsskridt.
    nstart = 0
    nmax = int(tmax/Dt) # hvor mange tidsskridt skal der laves, givet varighed Dt og maksimum tmax?
    nn = range(nstart,nmax)

    # vi laver en tom array der skal indeholde vores x værdier (fjederens position) for hvert tidsskridt.
    # den skal være lige så lang som vores tids-array tt.
    #For at lave en 'tom' array (fyld med 0'er) med en vis længde bruger jeg:
    xx = np.zeros(len(nn))

    # jeg skal også bruge en lignende array for hastighed og acceleration i hvert tidsskridt.
    vv = np.zeros(len(nn))
    aa = np.zeros(len(nn))

    # vi indsætter vores startværdier for sted og hastighed
    xx[0] = startværdier[0]
    vv[0] = startværdier[1]

    # vi beregner startaccelerationen ifølge hookes' lov (fjederkraftloven) og dæmpningsloven
    m = parametre[0]
    k = parametre[1]
    b = parametre[2]
    aa[0] = (-k/m)*(xx[0])-(b/m)*(vv[0]**v_potens)
    print(aa[0])

    # Nu laver jeg et loop over tidsskridtene. Hver gang n stiger, beregnes der en ny
    # værdi af x, v og acceleration. Disse bruges så til at bestemme værdiene for det
    # næste tidsskridt, og så videre.

    for n in nn:
        # print nuværende værdier for dette tidsskridt
        #print('skridt:',n,'tid',round(n*Dt,3),'x:',round(xx[n],3),
        #      'v',round(vv[n],3),'a:',round(aa[n],3))
        # beregn sted, hastighed og acceleration for næste tidsskridt
        (xud,vud,aud) = tidsskridt(xx[n],vv[n],aa[n],Dt,parametre,v_potens)
        # gem disse værdier i vores xx, vv og aa arrays
        if n < nmax-1: # tjekker at vi ikke er ved den sidste beregning (fordi vores arrays har længde nmax)
            xx[n+1] = xud
            vv[n+1] = vud
            aa[n+1] = aud

    # nu er vi færdig med at loope over n, og vores arrays er fyldt ud.
    # Så skal vi bare plotte vores sted- og hastighedsfunktioner.
    # jeg starter med at lave en array, der giver *tidspunktet* til hvert tidsskridt.
    tt = [n*Dt for n in nn]
    # så sender jeg data til noget plotte kode, gemt væk i en separat funktion.
    smukfigur(xx,vv,aa,tt,parametre,v_potens)
    # så er vi færdige!

def tidsskridt(xin,vin,ain,Dt,parametre,v_potens):
    '''
    givet værdier for x, v og a i tidsskridt n, beregner værdier i tidsskridt n+1.
    '''
    m = parametre[0]
    k = parametre[1]
    b = parametre[2]
    # forst beregnes den nye hastighed, v(n+1). den afhænger af den nuværende hastighed samt
    # den nuværende acceleration.
    # Denne beregning forudsætter at accelerationen er konstant over varigheden Dt.
    vud = vin + ain*Dt
    # så beregnes det nye sted, x(n+1), nu udfra approksimationen at v er konstant over Dt.
    xud = xin + vin*Dt
    # til sidst beregnes den nye kraft, udfra hookes' lov og dæmpningsloven.
    # hvis v_afhængighed = 1 så beregner den ma = -kx -bv
    # hvis v_afhængighed = 2 så beregner den ma = -kx -b(v^2)
    # leddet 'sign(vud)' kombineret med brug af 'abs(vud)'
    # sørger for at kraften altid peger modsat bevægelsesretningen,
    # selvom der er et v^n led.
    aud = (-k/m)*xud-(b/m)*np.sign(vud)*float(abs(vud))**v_potens
    # disse tre tal skal returneres - de svarer til sted,
    # hastighed og acceleration i tidsskridt n+1.
    return (xud,vud,aud)

def smukfigur(xx,vv,aa,tt,parametre,v_potens):
    '''
    tegner nogle figurer ud fra vores arrays.
    '''
    plt.figure()
    plt.plot(tt,xx)
    plt.xlabel('Tid [sekunder]')
    plt.ylabel('Sted [meter]')
    plt.figtext(0.5,0.85,'Fjederkonstant: '+str(parametre[1])+' N/m')
    plt.figtext(0.5,0.82,'Masse: '+str(parametre[0])+' kg')
    plt.figtext(0.5,0.79,'Dæmpningskoefficient: '+str(parametre[2]))
    plt.figtext(0.5,0.76,'Hastigheds afhængighed: v^'+str(v_potens))
    plt.figure()
    plt.plot(tt,vv)
    plt.xlabel('Tid [sekunder]')
    plt.ylabel('Hastighed [m/s]')
    plt.figtext(0.5,0.85,'Fjederkonstant: '+str(parametre[1])+' N/m')
    plt.figtext(0.5,0.82,'Masse: '+str(parametre[0])+' kg')
    plt.figtext(0.5,0.79,'Dæmpningskoefficient: '+str(parametre[2]))
    plt.figtext(0.5,0.76,'Hastigheds afhængighed: v^'+str(v_potens))
    plt.show()
    plt.plot(tt,aa)
    plt.xlabel('Tid [sekunder]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.figtext(0.5,0.85,'Fjederkonstant: '+str(parametre[1])+' N/m')
    plt.figtext(0.5,0.82,'Masse: '+str(parametre[0])+' kg')
    plt.figtext(0.5,0.79,'Dæmpningskoefficient: '+str(parametre[2]))
    plt.figtext(0.5,0.76,'Hastigheds afhængighed: v^'+str(v_potens))
    plt.show()

def def_start():
    '''
    Returnerer en start position og start hastighed, hvis brugeren ikke gider inputte dem.
    '''
    # returnerer (x_0,v_0)
    return(1.,0.)

def def_param():
    '''
    Returnerer masse, fjederkonstant og dæmpningskoefficient, hvis brugeren ikke gider inputte.
    '''
    return(1.,0.5,0.25)
