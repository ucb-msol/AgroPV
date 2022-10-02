###########################################################################################
# Modeling and Simulation of AgroPV Greenhouse Solar Energy Absorption
# Copyright 2022 Roger Isied, Emre Mengi, Tarek Zohdi. All rights reserved.
# For Authorized Use Only.
###########################################################################################

#%% Importing packages/visualization settings and functions

import numpy as np
import copy
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
plt.rcParams.update({'legend.fontsize': 15})
from matplotlib import rc
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams.update({'font.size': 22})

## ONLY IMPORT BELOW IF VISUALIZING WITH TECPLOT
import logging as log

log.basicConfig(level=log.INFO)

import tecplot as tp
from tecplot.session import set_style
from tecplot.constant import ColorMapDistribution
from tecplot.constant import *
import sys
### Begin WSF ###
is_batch = True if "batch" in sys.argv else False
### End WSF ###

#pysolar imports
import pysolar.solar as ps
import datetime
import pytz

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# %% Importing TecPlot Functions

def create_animation(outfile, plot, nframes):
    """
    Using the tp.export.animation_mpeg4() context manager, the F-18 is
    recorded doing an "aileron roll" by rotating and translating the
    viewer with respect to the data by a small amount and capturing
    each frame of the animation with a call to ani.export_animation_frame()
    """
    with tp.session.suspend():
        opts = dict(
            width=400,
            animation_speed=30,
            supersample=3,
        )
        # view = plot.view
        # plot.translate_view(view, -15)
        with tp.export.animation_mpeg4(outfile, **opts) as ani:
          for i in range(nframes):
            # view.rotate_axes(5, (1, 0, 0))
            # translate_view(view, 30 / args.nframes)
            ani.export_animation_frame()

#%% Solar simulation function
def greenHouseSim(Lam, consts):
    
    '''
    greenHouseSim. Simulates ray tracing from initial input angle based on sunlight data. Tracks rays reflecting and
    refracting off of greenhouse as well as rays contacting ground within greenhouse
    
    Controllable Inputs:
    *Lam (Sx1 Double) -- Input Design String with control parameters
     - gamma (1x1 Double) -- Greenhouse tranlucency between 0 (fully opaque) and 1 (fully transparent)
     - ns (1x1 Double) -- Greenhouse refractive index
     - p1 (1x1 Double) -- Greenhouse contour e1 direction exponent
     - p3 (1x1 Double) -- Greenhouse contour e3 direction exponent
     - a (1x1 Double) -- Greenhouse contour curvature amplitude in e1 direction
     - b (1x1 Double) -- Greenhouse contour curvature amplitude in e2 direction
     - w1 (1x1 Double) -- Greenhouse contour curvature frequency in e1 direction
     - w2 (1x1 Double) -- Greenhouse contour curvature frequency in e2 direction
     
    *Consts (26x1 Double) -- Input constants for given green house system 
     - latitude (1x1 Double) -- latitude of location on earth (degrees)
     - longitude (1x1 Double) -- longitude of location on eath (degrees)
     - timezone (1x1 DstTzInfo) -- pysolar timezone input
     - year (1x1 Int) -- year of simulation
     - month (1x1 Int) -- month of simulation
     - day (1x1 Int) -- day of simulation
     - ng (1x1 Double) -- Refractive index of ground (depends on vegetation in greenhouse)
     - p2 (1 x 1 Double) -- Greenhouse contour e2 direction exponent (restricting to single cross sectioned green house)
     - sThick (1x1 Double) -- Thickness of solar panel (m)
     - Nr (1x1 Integer) -- Number of light rays
     - c (1x1 Double) -- Nominal speed of light in a vacuum (m/s)
     - Pfrac (1x1 Double) -- Fraction between 0 and 1 indicating percentage of original light ray power below which ray will stop being tracked
     - R1 (1x1 Dobule) -- Greenhouse radius along e1 direction (m)
     - R2 (1x1 Dobule) -- Greenhouse radius along e2 direction (m)
     - R3 (1x1 Dobule) -- Greenhouse radius along e3 direction (m)
     - rayLength1 (1x1 Double) -- Length of square ray beam in e1 direction (m)
     - rayLength2 (1x1 Double) -- Length of square ray beam in e2 direction (m)
     - domLim (1x1 Double) -- Domain limit in all dimensions (m)
     - genDist (1x1 Double) -- Radius of initial position of rays (m)
     - eta (1x1 Double) -- Solar panel efficiency between 0 and 1
     - etaG (1x1 Double) -- Photosynthesis efficiency between 0 and 1
     - weight1 (1x1 Double) Solar term scaling factor for cost function
     - weight2 (1x1 Double) Ground power term scaling factor for cost function
     - Pinit (1x1 Double) -- Solar Ray irrdadiance when it initially reaches greenhouse (W/m^2)
     - altRad (1x1 Double) -- Altitude angle of incoming rays (rad)
     - aziRad (1x1 Double) -- Azimuth angle of incoming rays (rad)
    '''

    # Extracting design parameters from design vector
    gamma = Lam[0]
    ns = Lam[1]
    p1 = Lam[2]
    p3 = Lam[3]
    a = Lam[4]
    b = Lam[5]
    w1 = Lam[6]
    w2 = Lam[7]
    
    # Extracting constants from constant array
    
    ng = consts[6]
    p2 = consts[7]
    sThick = consts[8]
    Nr = int(consts[9])
    c = consts[10]
    Pfrac = consts[11]
    R1 = consts[12]
    R2 = consts[13]
    R3 = consts[14]
    domLim = consts[15]
    genDist = consts[16]
    eta = consts[17]
    etaG = consts[18]
    weight1 = consts[19]
    weight2 = consts[20]
    Pinit = consts[21]
    altRad = consts[22]
    aziRad = consts[23]

    ## Determine material properties of solar panel & ground ##
    # ASSUMING NO MAGNETIC VARIANCE
    # APPROXIMATING REFRACTIVE INDEX OF AIR TO BE 1
    nhat = ns # Calculate refractive index for panel
    nhatG = ng # Calculate refractive index for ground

    # Define solar ray region based on defined domain limit
    xMin = -genDist
    xMax = genDist
    yMin = -genDist
    yMax = genDist
    # R = (0.98*domLim)/np.sin(altRad)
    R = 0.98*domLim
    
    Ptol = Pfrac*Pinit # Power tolerance for which below we stop tracking rays
        
    Pr = Pinit*np.ones((Nr,1))
    
    ## Initial Ray Position (initial position need to be translated to ensure hitting the surface at all angles)
    ray_pos_init = R * np.array([np.cos(altRad)*np.sin(aziRad), np.cos(altRad)*np.cos(aziRad), np.sin(altRad)])
    
    # set initial ray position depending on sun azi/alt                  
    pos = np.array([(xMax-xMin)*np.random.rand(Nr)+ xMin + ray_pos_init[0], \
                    (yMax-yMin)*np.random.rand(Nr)+ yMin + ray_pos_init[1],np.ones(Nr)*ray_pos_init[2]]) # Initial Ray Positions  
  
    pos = pos.T

    # Initial ray velocities based on incoming angle w.r.t. e1-axis
    vel = np.array([-c*np.cos(altRad)*np.sin(aziRad)*np.ones(Nr), -c*np.cos(altRad)*np.cos(aziRad)*np.ones(Nr), -c*np.sin(altRad)*np.ones(Nr)]) # Initial Ray Velocities
    vel = vel.T
    
    prePrepos = copy.deepcopy(pos) # position from 2 time step previous
    prevPos = copy.deepcopy(pos)

    # Initialize position of moving and impacted rays
    posTot = list()
    velTot = list()
    gTot = list()
    sTot = list()
    
    # dt = 0.01*(ray_pos_init[2]/c) # Time step based on velocity and initial height of rays
    dt = 0.01*(5/c)

    sAbs = 0 # Initialize power absorbed by solar panel
    gAbs = 0 # Initialize power absorbed by ground

    Active = np.array([True for i in range(Nr)]) # Logical array to indicate what rays are still in flight

    ts = 0 # Initialize number of time steps
    
    # Initialzing lists to check which rays are inside and outside green house
    PPinHouse = [np.array([], dtype=np.int64)]
    PPinAir = [np.array([], dtype=np.int64)]
    
    # Initialzing lists to check which rays are inside and outside green house
    PinHouse = [np.array([], dtype=np.int64)]
    PinAir = [np.array([], dtype=np.int64)]
    
    inHouse = [np.array([], dtype=np.int64)]
    inAir = [np.array([], dtype=np.int64)]
    
    # Check envelope equation for initial time step
    rayContour = ((np.abs(pos[:,0]))/R1)**p1 + ((np.abs(pos[:,1]))/R2)**p2 + ((np.abs(pos[:,2]))/R3)**p3 + a*np.sin(w1*pos[:,0]) + b*np.sin(w2*pos[:,1])
    # determine where rays exist in initial time step
    inHouse = list(np.where((rayContour <= 1))) # Indices of rays that are located inside boundary of green house
    inGround = list(np.where((pos[:,2] <= 0))) # Determine which rays have impacted the ground
    inAir = list(np.where((rayContour > 1))) # Indices of rays that are located inside boundary of solar
    
    # Set counter for number of contacting rays (for cost function)
    numContRays = 0
    
    # Initialize generated electric power and plant absorbed power
    Pelectric = 0
    Pphotosynth = 0
    
    # Initialize time step scaling as 1 so no change in time step size
    dtScale = 1

    while np.any(Active) and ts < 500 and Nr < 100000: # While any of the rays are still traveling and we haven't duplicated too many rays
        
        
        indAct = np.where(Active) # Determine indices of active rays
        
        # Indices of rays moving from air to green house
        indAirToHouse = list(set(indAct[0][:]) & set(inHouse[0][:]) & set(PinAir[0][:]) & set(PPinAir[0][:]))
        
        
        # Indices of rays moving from greenhouse to air
        indHouseToAir = list(set(indAct[0][:]) & set(inAir[0][:]) & set(PinHouse[0][:]) & set(PPinHouse[0][:]))

        # Indices of arrays out of solar region and impacted ground
        houseGround = list(set(inGround[0][:]) & set(indAct[0][:]) & set(inHouse[0][:]))
        airGround = list(set(inGround[0][:]) & set(indAct[0][:]) & set(inAir[0][:]))
         
        numContRays += np.size(indAirToHouse)

        if np.size(indAirToHouse) > 0: # If any ray has impacted the solar panel from the outside
            
            # positions of rays in contact with greenhouse
            contactX = pos[indAirToHouse,0]
            contactY = pos[indAirToHouse,1]
            contactZ = pos[indAirToHouse,2]
                        
            # Calculate gradient of panel surface at impact point
            gradF = np.array([(contactX*p1*(np.abs(contactX)**(p1-2)))/(np.abs(R1)**p1) + a*w1*np.cos(w1*contactX),
                               (contactY*p2*(np.abs(contactY)**(p2-2)))/(np.abs(R2)**p2) + b*w2*np.cos(w2*contactY),
                               ((contactZ)*p3*(np.abs((contactZ))**(p3-2)))/(np.abs(R3)**p3)])


            # Normalize gradient to get normal vector
            normal = gradF/np.linalg.norm(gradF,2,0)

            # Calculate angle of incidence using velocity of solar rays and normal vector to panel surface
            thetai = np.array(np.arccos(np.sum(vel[indAirToHouse,:]*normal.T,1)/(c)))
            thetai = np.reshape(thetai,[thetai.size,1])
            
            # Calculate refractive angle
            thetaRef = np.arcsin(np.sin(thetai)/ns)

            # Calculate reflectivity of impacted rays on solar panel
            
            term1 = np.abs(np.cos(thetai))
            term2 = (nhat**2 - np.sin(thetai)**2)**0.5
            
            Reflect = np.array(np.abs(np.sin(thetai)/nhat) < 1.)* \
                np.array(0.5*(((nhat*term1 - np.abs(term2)/(nhat*term1 + np.abs(term2))**2 + \
                         ((term1 - term2)/(term1 + term2))**2)))) + np.array(np.abs(np.sin(thetai)/nhat) >= 1.)*1.


            # add velocity to refracted rays
            vel = np.vstack((vel,vel[indAirToHouse,:]))
            
            vel[indAirToHouse,:] -= 2*(c*normal.T*np.cos(thetai)) # Update velocity of reflected rays
            
            # Append new refracted beams to position array (will be same as impact position)
            pos = np.vstack((pos,pos[indAirToHouse,:] + np.hstack((sThick*np.tan(thetaRef),np.zeros((thetaRef.size,2))))))

            Pabs = Pr[indAirToHouse] - Reflect*Pr[indAirToHouse] # Power absored by green house
            
            Ppanel = (1-gamma)*Pabs # Power absored by solar panel
            
            Pelectric += np.sum(eta*Ppanel,0) # Power converted to electricity (sum contribution from each light ray)
            
            Pr = np.vstack((Pr,Ppanel*gamma)) # Append the power of refracted beams to power array

            Pr[indAirToHouse] *= Reflect # Update power remaining in reflected rays
            
            Active = np.hstack((Active,np.array([True]*np.size(indAirToHouse)))).T
            
        if np.size(indHouseToAir) > 0: # If any ray has impacted the solar panel from the inside
            
            # positions of rays in contact with greenhouse
            contactX = pos[indHouseToAir,0]
            contactY = pos[indHouseToAir,1]
            contactZ = pos[indHouseToAir,2]
                        
            # Calculate gradient of panel surface at impact point
            gradF = np.array([(contactX*p1*(np.abs(contactX)**(p1-2)))/(np.abs(R1)**p1) + a*w1*np.cos(w1*contactX),
                               (contactY*p2*(np.abs(contactY)**(p2-2)))/(np.abs(R2)**p2) + b*w2*np.cos(w2*contactY),
                               ((contactZ)*p3*(np.abs((contactZ))**(p3-2)))/(np.abs(R3)**p3)])


            # Normalize gradient to get normal vector
            normal = gradF/np.linalg.norm(gradF,2,0)

            # Calculate angle of incidence using velocity of solar rays and normal vector to panel surface
            thetai = np.array(np.arccos(np.sum(vel[indHouseToAir,:]*normal.T,1)/(c)))
            thetai = np.reshape(thetai,[thetai.size,1])
            
            # Calculate refractive angle
            thetaRef = np.arcsin(np.sin(thetai)/ns)

            # Calculate reflectivity of impacted rays on solar panel
            
            term1 = np.abs(np.cos(thetai))
            term2 = (nhat**2 - np.sin(thetai)**2)**0.5
            
            Reflect = np.array(np.abs(np.sin(thetai)/nhat) < 1.)* \
                np.array(0.5*(((nhat*term1 - term2)/(nhat*term1 + term2))**2 + \
                         ((term1 - term2)/(term1 + term2))**2)) + np.array(np.abs(np.sin(thetai)/nhat) >= 1.)*1.
            
            # add velocity to refracted rays
            vel = np.vstack((vel,vel[indHouseToAir,:]))

            vel[indHouseToAir,:] -= 2*(c*normal.T*np.cos(thetai)) # Update velocity of reflected rays
            
            # Append new refracted beams to position array (will be same as impact position)
            pos = np.vstack((pos,pos[indHouseToAir,:] + np.hstack((sThick*np.tan(thetaRef),np.zeros((thetaRef.size,2))))))

            Pabs = Pr[indHouseToAir] - Reflect*Pr[indHouseToAir] # Power absored by green house
            
            Ppanel = (1-gamma)*Pabs # Power absored by solar panel
            
            Pelectric += np.sum(eta*Ppanel,0) # Power converted to electricity (sum contribution from each light ray)
            
            Pr = np.vstack((Pr,Ppanel*gamma)) # Append the power of refracted beams to power array

            Pr[indHouseToAir] *= Reflect # Update power remaining in reflected rays
            
            Active = np.hstack((Active,np.array([True]*np.size(indHouseToAir)).T))

        if np.size(houseGround) > 0: # If any ray has impacted the ground (ONLY DOING REFLECTIONS HERE)

            normalG = np.array([[0,0,-1]]) # Define constant normal vector (assuming flat ground)

            # Calculate angle of incidence using velocity of solar rays and normal vector to ground
            thetai = np.array(np.arccos(np.sum(vel[houseGround,:]*normalG,1)/c))
            thetai = np.reshape(thetai,[thetai.size,1])
            
            # Calculate reflectivity of impacted rays on ground
            
            term1 = np.cos(thetai)
            term2 = (nhatG**2 - np.sin(thetai)**2)**0.5
            
            gReflect = np.array(np.abs(np.sin(thetai)/nhatG) < 1.)* \
                np.array(0.5*(((nhatG*term1 - term2)/(nhatG*term1 + term2))**2 + \
                         ((term1 - term2)/(term1 + term2))**2)) + np.array(np.abs(np.sin(thetai)/nhatG) >= 1.)*1.
            
            vel[houseGround,:] -= 2*(c*np.cos(thetai)*normalG) # Update velocity of reflected rays

            PabsGround = Pr[houseGround] - gReflect*Pr[houseGround] # Calculate power absored by ground
            
            Pphotosynth += np.sum(etaG*PabsGround,0)

            Pr[houseGround] *= gReflect # Update power remaining in reflected rays

        # Determine which rays have left the domain or have reduced below the threshold power level
        powRem = np.array(np.where(Pr < Ptol))
        domRem = np.array((np.abs(pos[:,0]) > domLim) | (np.abs(pos[:,1]) > domLim) | (np.abs(pos[:,2]) > domLim))
        if domRem.size > 0 and numContRays > 0: # If rays have left the domain
            Active[domRem] = False # Set impacted rays to false

        if powRem.size > 0: # If rays have reduced below power tolerance
            Active[powRem] = False # Set impacted rays to false
            
        if np.size(airGround) > 0: # If rays hit ground outside of the green house
            Active[airGround] = False # Set impacted rays to false

        # Save positions of active and not active rays for plotting 
        posTot.append(pos[Active,:])
        velTot.append(vel[Active,:])

        # Append ground impacted rays to list
        if np.size(gTot) == 0:
            gTot.append(pos[list(set(inGround[0][:]) & set(inHouse[0][:])),:])
        else:
            gTot.append(np.vstack([np.array(gTot[-1]),pos[list(set(inGround[0][:]) & set(inHouse[0][:])),:]]))

        # Append greenhouse impacted rays to list
        if np.size(sTot) == 0:
            sTot.append(np.vstack([pos[indAirToHouse,:],pos[indHouseToAir,:]]))
        else:
            sTot.append(np.vstack([np.array(sTot[-1]),pos[indAirToHouse,:],pos[indHouseToAir,:]]))

        # Update number of rays
        Nr = pos[Active,:].shape[0]
                
        # print("Number of Active Rays: ",Nr)
        
        pos[Active,:] += dt*vel[Active,:] # Update new posiions of rays
        
        # Saving previous two contour values to distinguish between reflections and refractions
        PPinHouse = PinHouse
        PPinAir = PinAir
        
        PinHouse = inHouse
        PinAir = inAir
        
        # Check envelope equation for current time stepfor
        rayContour = ((np.abs(pos[:,0]))/R1)**p1 + ((np.abs(pos[:,1]))/R2)**p2 + ((np.abs(pos[:,2]))/R3)**p3 + a*np.sin(w1*pos[:,0]) + b*np.sin(w2*pos[:,1])
        # determine where rays exist in current time step
        inHouse = list(np.where((rayContour < 1))) # Indices of rays that are located inside boundary of green house
        inGround = list(np.where((pos[:,2] <= 0))) # Determine which rays have impacted the ground
        inAir = list(np.where((rayContour >= 1))) # Indices of rays that are located inside boundary of solar
        
        ts = ts + 1 # Iterate Number of time steps
    # Scaling 
    Po = Pinit*numContRays # Initial power from rays that come into contact with greenhouse
    
    PoSolar = (1/3)*Po # Want solar to get ~33% of incoming light that comes in contact with green house
    PoGreenHouse = (1/6)*Po # Want plants to get ~16% of incoming light that comes in contact with green house
    
    if numContRays == 0: # if no rays hit the green house
        Pi = weight1 + weight2
        alpha = 1
        gamma = 1
    else: 
        alpha = np.abs((PoSolar-Pelectric)/(PoSolar))
        gamma = np.abs(((PoGreenHouse-Pphotosynth)/(PoGreenHouse)))
        
        Pi = weight1*alpha + weight2*gamma # Cost function
    
    return(Pi, alpha, gamma, posTot, gTot, sTot, ts, velTot)

#%% Define Day simulation function

def DaySim(Lam,Func,consts,latitude,longitude,timezone,year,month,day):
    
    '''
    DaySim(Lam,Func,consts,latitude,longitude,timezone,year,month,day)
    Runs greenhouse simulator for each hour throughout the inputted day and location based
    on when the sun is above the horizon.
    '''
    
    hourCounter = 0 # counter to determine how many hours were considered
    piTot = 0 # contains the sum of all of the costs from every considered hour
    alphaTot = 0
    gammaTot = 0
    
    
    # Initialize position of moving and impacted rays
    dayPosTot = list()
    dayVelTot = list()
    dayGTot = list()
    daySTot = list()
    
    dayTs = 0 # initialize number of time steps in full day
    
    dayTsTot = list() # Initialize array to cary hour data for each time step
    
    for daytime in range(24):
        
        date = datetime.datetime(year, month, day, hour = daytime, minute=0, second=0,microsecond=0, tzinfo=timezone)
        altDeg = ps.get_altitude(latitude, longitude, date)
        aziDeg = ps.get_azimuth(latitude, longitude, date)
        altRad = altDeg * np.pi / 180 # Altitude angle of incoming rays (rad)
        aziRad = aziDeg * np.pi / 180 # Azimuth angle of incoming rays (rad)
        
        if altRad > 0 and altRad < np.pi: #if altitude angle is greater than 0 (sun is above horizon)
            
            hourCounter += 1 #add to considered hour
            Pinit = ps.radiation.get_radiation_direct(date,altDeg) # Solar ray irradiance for given time and location (W/m^2)
            constsInput = np.hstack((consts,Pinit,altRad,aziRad)) # append to constants to be used in greenhouse sim function
            Pi, alpha, gamma, posTot, gTot, sTot, ts, velTot = Func(Lam,constsInput)
            piTot += Pi
            alphaTot += alpha
            gammaTot += gamma
            
            dayTs += ts
            dayTsTot.extend(daytime*np.ones(ts))
                # Save positions of active and not active rays for plotting
            if posTot[0][:,0].size > 0:
                dayPosTot.extend(posTot)
                dayVelTot.extend(velTot)
                dayGTot.extend(gTot)
                daySTot.extend(sTot)
            
        # print("hour: " + str(daytime))
    
    piAvg = piTot/hourCounter # Average cost over a day
    alphaAvg = alphaTot/hourCounter
    gammaAvg = gammaTot/hourCounter
        
    return(piAvg,alphaAvg,gammaAvg,dayPosTot,dayGTot,daySTot,dayTs,dayVelTot,dayTsTot)
            
    

#%% Define GA function

def myGA(S,G,P,K,SB,Func,consts):
    
    '''
    myGA(S,G,P,K,SB,Func,consts)
    
    Runs the genomic based optimizer for the inputed cost function calculated by Func, process
    constants, for S number of initial design strings, G generations, P parents and K
    children per generation (both P and K must be even numbers given this scheme).
    The search bounds for each design parameter is defined in SB
    '''
    
    # importing constants for daysim function
    latitude = consts[0]
    longitude = consts[1]
    timezone = consts[2]
    year = consts[3]
    month = consts[4]
    day = consts[5]
    
    # Initialize all variables to be saved
    Min = np.zeros(G) # Minimum cost for each generation
    PAve = np.zeros(G) # Parent average for each generation
    Ave = np.zeros(G) # Total population average for each generation
    
    Pi = np.zeros(S) # All costs in an individual generation
    
    alpha = np.zeros(S) # All solar panel losses ratios in each generation
    gamma = np.zeros(S) # All ground absorption ratios in each generation
    
    alphaMin = np.zeros(G) #solar panel losses ratio associated with best cost for each generation
    gammaMin = np.zeros(G) # ground absorption ratio associated with best cost for each generation
    
    alphaPAve = np.zeros(G) # solar panel losses ratio for top parents for each generation
    gammaPAve = np.zeros(G) # Aground absorption ratio value for top parents for each generation
    
    alphaAve = np.zeros(G) # Agerage solar panel losses ratio for whole population for each generation
    gammaAve = np.zeros(G) # Average ground absorption ratio for whole population for each generation
    
    
    # Generate initial random population
    Lam = np.array([(SB[i,1]-SB[i,0])*np.random.rand(S,1) + SB[i,0] for i in range(SB.shape[0])])[...,0].T
        
    # In first generation, calculate cost for all strings.
    # After, only calculate new strings since fitness for top P parents already calculated
    start = 0 
    
    for i in range(G): # Loop through generations
        
        
        # Calculate fitness of unknown design string costs
        for j in range(start,S): # Evaluate fitness of strings
            
            # Plug in design string control variables and array of function constants
            # output = Func(Lam[j,:], consts) # Outputs tuple of function outputs
            output = Func(Lam[j,:],greenHouseSim,consts,latitude,longitude,timezone,year,month,day)
            
            Pi[j] = output[0] # Extract cost from tuple outputs and assign it to cost array
            alpha[j] = output[1]
            gamma[j] = output[2]
            
        
        # Sort cost and design strings based on performance
        ind = np.argsort(Pi)
        Pi = np.sort(Pi)
        alpha = alpha[ind]
        gamma = gamma[ind]
        Lam = Lam[ind,:]
        
        # Generate offspring radnom parameters and indices for vectorized offspring calculation
        phi = np.random.rand(K,SB.shape[0]) # Generate random weights for offspring
        ind1 = range(0,K,2) # First set of children based on even numbered parameters
        ind2 = range(1,K,2) # Second set of children based on odd numbered parameters
        
        Parents = Lam[0:P,:] # Top P performing parents
        Children1 = phi[ind1,:]*Lam[ind1,:] + (1-phi[ind1,:])*Lam[ind2,:] # First set of children
        Children2 = phi[ind2,:]*Lam[ind2,:] + (1-phi[ind2,:])*Lam[ind1,:] # Second set of children
        
        newPop = np.array([(SB[i,1]-SB[i,0])*np.random.rand(S-P-K,1) + SB[i,0] for i in range(SB.shape[0])])[...,0].T # New random population
        
        Lam = np.vstack((Parents, Children1, Children2, newPop)) # Stack parents, children, and new strings to use in next generation    
        
        # Save cost values for plotting
        Min[i] = Pi[0]
        PAve[i] = np.mean(Pi[0:P])
        Ave[i] = np.mean(Pi)
        
        alphaMin[i] = alpha[0]
        gammaMin[i] = gamma[0]

        alphaPAve[i] = np.mean(alpha[0:P])
        gammaPAve[i] = np.mean(gamma[0:P])
        
        alphaAve[i] = np.mean(alpha)
        gammaAve[i] = np.mean(gamma)
        
        # Update start to P such that only new string cost values are calculated
        start = P
        
        # Print miminum value of cost for debugging (should monotonically decrease over generations)
        print("Best cost for generation " + str(i+1) + ": " + str(Min[i]))
        
    bestLam = Lam[0,:] # Extracting best design string parameters afer all generations are run
    
    return(Lam, bestLam, Pi, Min, PAve, Ave, alphaMin, gammaMin, alphaPAve, gammaPAve, alphaAve, gammaAve)


#%% Numerical Example

# Search bounds for parameter X in format (X-,X+) in for variables given below
SB = np.array([[0.25,1], [2,5], [1,20], [1,20],[0,1.75],[0,1.75],[0,10],[0,10]])
# SB = np.array([[0.35373,0.35373], [2.08294,2.08294], [12.4543,12.4543], [17.6631,17.6631],[0.268633,0.268633],[1.4583,1.4583],[6.91489,6.91489],[8.73698,8.73698]])

'''
- gamma (1x1 Double) -- Greenhouse tranlucency between 0 (fully opaque) and 1 (fully transparent)
- ns (1x1 Double) -- Greenhouse refractive index
- p1 (1x1 Double) -- Greenhouse contour e1 direction exponent
- p3 (1x1 Double) -- Greenhouse contour e3 direction exponent
- a (1x1 Double) -- Greenhouse contour curvature amplitude in e1 direction
- b (1x1 Double) -- Greenhouse contour curvature amplitude in e2 direction
- w1 (1x1 Double) -- Greenhouse contour curvature frequency in e1 direction
- w2 (1x1 Double) -- Greenhouse contour curvature frequency in e2 direction
'''

ng = 1.4 # Ground refractive index (lettuce)
p2 = 20 # Greenhouse contour e2 direction exponent
R1 = 0.5 # Greenhouse radius along e1 direction (m)
R2 = 5 # Greenhouse radius along e2 direction (m)
R3 = 0.5 # Greenhouse radius along e3 direction (m)
Nr = 500 # Number of light rays
c = 3E8 # Speed of light in a vacuum (m/s)
genDist = 0.5 # side length of square beam of rays (m)
domLim = 1.5 # Limit of domain in e1 and e2 domain (m)
Pfrac = 0.05 #fraction of total power at which we stop tracking a ray
sThick = 0.1 # Thickness of solar panel (m)
eta = 1 # Solar panel efficiency
etaG = 1 # Photosynthesis efficiency (see https://www.fao.org/3/w7241e/w7241e05.htm#1.2.1)
weight1 = 2 # Solar term scaling factor for cost function
weight2 = 1 # Ground power term scaling factor for cost function

#pysolar function variables for Berkeley, CA
latitude = 37.8715
longitude = -122.2730
timezone = pytz.timezone('US/Pacific')
year = 2021
month = 7
day = 1

# Defining constants array for solar function in the order of the following variables

consts = np.array([latitude, longitude, timezone, year, month, day, ng, p2, sThick, Nr, c, Pfrac, R1, R2, R3, domLim, genDist, eta, etaG, weight1, weight2])

#%% Run GA optimizer

## Genetic Algorithm Parameters
K = 6 # Strings generated by breeding
P = 6 # Surviving strings for breeding
S = 20 # Design strings per generation
G = 200 # Total Generations

Lam, bestLam, Pi, Min, PAve, Ave, alphaMin, gammaMin, alphaPAve, gammaPAve, alphaAve, gammaAve = myGA(S,G,P,K,SB,DaySim,consts)

# #%% Save output data (for Nr500 run stuff)
# import os

# # Get the current working directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# np.save('Nr500 Runs/10/bestLam10',bestLam)

# np.save('Nr500 Runs/10/alphaAve10',alphaAve)
# np.save('Nr500 Runs/10/alphaMin10',alphaMin)
# np.save('Nr500 Runs/10/alphaPAve10',alphaPAve)

# np.save('Nr500 Runs/10/Min10',Min)
# np.save('Nr500 Runs/10/Ave10',Ave)
# np.save('Nr500 Runs/10/PAve10',PAve)

# np.save('Nr500 Runs/10/gammaPAve10',gammaPAve)
# np.save('Nr500 Runs/10/gammaAve10',gammaAve)
# np.save('Nr500 Runs/10/gammaMin10',gammaMin)

#%% Save output data and plot (for Sensitivity Study) 

# import os

# # Get the current working directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# np.save('Sensitivity Study/Nr10000/bestLam10000',bestLam)

# np.save('Sensitivity Study/Nr10000/alpha10000',alpha)

# np.save('Sensitivity Study/Nr10000/Pi10000',Pi)

# np.save('Sensitivity Study/Nr10000/gamma10000',gamma)


# fig = plt.figure(figsize = (15,7))
# plt.plot(range(S),Pi,label = r'$\Pi(\mathbf{\Lambda}^i):~\sigma = $ ' + str(round(np.std(Pi),3)))
# plt.plot(range(S),alpha,label = r'$\alpha(\mathbf{\Lambda}^i):~\sigma = $ ' + str(round(np.std(alpha),4)))
# plt.plot(range(S),gamma,label = r'$\gamma(\mathbf{\Lambda}^i):~\sigma = $ '+ str(round(np.std(gamma),4)))
# plt.plot(np.where(Pi == np.min(Pi))[0],np.min(Pi),'.',ms = 15,
#          label = r'$\Pi(\mathbf{\Lambda}^i)_{min} =$ '+ str(round(np.min(Pi),3)))
# plt.plot(np.where(Pi == np.max(Pi))[0],np.max(Pi),'.',ms = 15,
#          label = r'$\Pi(\mathbf{\Lambda}^i)_{max} =$ '+ str(round(np.max(Pi),3)))
# plt.plot(np.where(Pi == find_nearest(Pi,np.mean(Pi))),np.mean(Pi),'.',ms = 15,
#          label = r'$\Pi(\mathbf{\Lambda}^i)_{mean} =$ '+ str(round(np.mean(Pi),3)))
# plt.title(r'Performance Metrics under same design parameters for $N_r =$'+ str(Nr))
# plt.xlabel('Test Run')
# plt.ylabel('Performance Metric')
# plt.legend()
# plt.savefig('Sensitivity Study/Nr10000/plot.png')
# plt.show()

#%% save output plots
fig = plt.figure(figsize = (15,7))
plt.plot(range(G),Min)
plt.title("Evolution of Best Cost per Generation")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.savefig('Nr500 Runs/10/best.png')
plt.show()



fig = plt.figure(figsize = (15,7))
plt.plot(range(G),PAve)
plt.title("Evolution of Average Parent Cost per Generation")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.savefig('Nr500 Runs/10/parent.png')
plt.show()

fig = plt.figure(figsize=(15,7))
plt.plot(range(G),Min,label='Total Cost')
plt.plot(range(G),alphaMin,label='Solar Penalty')
plt.plot(range(G),gammaMin,label='Plant Penalty')
plt.xlabel('Generation')
plt.ylabel('Cost Parameter Value')
plt.title('Best Cost Parameter Evolution')
#plt.legend(['$\Pi$', '$\alpha$','$\gamma$'])
# Put a legend to the right of the current axis
plt.legend(loc='center right')
#plt.legend(loc='lower left')
plt.savefig('Nr500 Runs/10/best_all.png')
plt.show()

fig = plt.figure(figsize=(15,7))
plt.plot(range(G),PAve,label='Total Cost')
plt.plot(range(G),alphaPAve,label='Solar Penalty')
plt.plot(range(G),gammaPAve,label='Plant Penalty')
plt.xlabel('Generation')
plt.ylabel('Cost Parameter Value')
plt.title('Parent Average Cost Parameter Evolution')
#plt.legend(['$\Pi$', '$\alpha$','$\gamma$'])
plt.legend(loc='center right')
plt.savefig('Nr500 Runs/10/parent_all.png')
plt.show()

#%% Plot light ray simulation on TecPlot

scale = 1 # scaling for time steps on animation (larger the scale, fewer time steps)
hour = 12 # hour in day for which light approaches green house
testLam = np.array([[0.06370442],[2.01284415],[4.58778782],[13.96279516],
                    [0.90911276],[1.72985394],[4.12462758],[7.33361111]])

test = 1 ## type of plotting (test == 0 is plotting single flash test, test == 1 plots best GA solution, test == 2 plots a full day best solution) 
if test == 0:
    # run test solution

    date = datetime.datetime(year, month, day, hour, minute=0, second=0,microsecond=0, tzinfo=timezone)
    altDeg = ps.get_altitude(latitude, longitude, date)
    aziDeg = ps.get_azimuth(latitude, longitude, date)
    altRad = altDeg * np.pi / 180 # Altitude angle of incoming rays (rad)
    aziRad = aziDeg * np.pi / 180 # Azimuth angle of incoming rays (rad)

    if altRad > 0 and altRad < np.pi: #if altitude angle is greater than 0 (sun is above horizon)
        
        Pinit = ps.radiation.get_radiation_direct(date,altDeg) # Solar ray irradiance for given time and location (W/m^2)
        constsInput = np.hstack((consts,Pinit,altRad,aziRad)) # append to constants to be used in greenhouse sim function
        Pi, alpha, gamma, posTot, gTot, sTot, ts, velTot = greenHouseSim(testLam,constsInput)
    
    bestLam = testLam
    tsTot = list(hour*np.ones(ts))
elif test == 1:
    bestLam = Lam[0,:]
    consts = np.array([latitude, longitude, timezone, year, month, day, ng, p2, sThick, Nr, c, Pfrac, R1, R2, R3, domLim, genDist, eta, etaG, weight1, weight2])
    date = datetime.datetime(year, month, day, hour, minute=0, second=0,microsecond=0, tzinfo=timezone)
    altDeg = ps.get_altitude(latitude, longitude, date)
    aziDeg = ps.get_azimuth(latitude, longitude, date)
    altRad = altDeg * np.pi / 180 # Altitude angle of incoming rays (rad)
    aziRad = aziDeg * np.pi / 180 # Azimuth angle of incoming rays (rad)
    Pinit = ps.radiation.get_radiation_direct(date,altDeg) # Solar ray irradiance for given time and location (W/m^2)
    constsInput = np.hstack((consts,Pinit,altRad,aziRad)) # append to constants to be used in greenhouse sim function
    Pi, alpha, gamma, posTot, gTot, sTot, ts, velTot = greenHouseSim(bestLam,constsInput)
    tsTot = list(hour*np.ones(ts))
elif test == 2:
    bestLam = Lam[0,:]
    # bestLam = testLam
    Pi,alphaAvg,gammaAvg, posTot, gTot, sTot, ts, velTot,tsTot = DaySim(bestLam,greenHouseSim,consts,latitude,longitude,timezone,year,month,day)

# Plotting grid parameters
N = 20
L = 3*genDist

x = np.linspace(-L,L,N) # Define grid points in one dimension
z = np.linspace(0,L,N)

X,Y,Z = np.meshgrid(x,x,z)

if not is_batch:
    tp.session.connect()


now = time.time()
tp.new_layout()
print(tp.version_info)
print(tp.sdk_version_info)
print("Time to initialize the Tecplot engine: ", time.time() - now)

now = time.time()

with tp.session.suspend(): # This apparently makes doing things in Tecplot faster
    
    ds = tp.active_frame().create_dataset('Light Rays', ['x','y','z','GreenHouse','u','v','w']) # Create 1 data set with all the variables you need
    
    shape = (N,N,N) # Every time you add a zone, you have to define its shape
    green_house = ds.add_ordered_zone('Green House',shape) # This adds a zone

    # Below is adding your python variables to the zone
    green_house.values('x')[:] = X.ravel()
    green_house.values('y')[:] = Y.ravel()
    green_house.values('z')[:] = Z.ravel()
    green_house.values('GreenHouse')[:] = (((np.abs(X.ravel()))/R1)**bestLam[2] + ((np.abs(Y.ravel()))/R2)**p2 + ((np.abs(Z.ravel()))/R3)**bestLam[3]) + bestLam[4]*np.sin(bestLam[6]*X.ravel()) + bestLam[5]*np.sin(bestLam[7]*Y.ravel())

    # For transient data, set the strand of all of the things you want to plot together the same and then set the time for each thing
    green_house.strand = 0 # Setting the strand to 0 makes it static so it won't be dependent on time
    green_house.solution_time = 0
    tp.active_frame().plot_type = tp.constant.PlotType.Cartesian3D # Use this to convert from a "sketch" to a 3D plot
    text = tp.active_frame().add_text(' ' , (50,80), size=32, anchor=TextAnchor.Center)

    plot = tp.active_frame().plot() # Define the plot you're working on
    cont = plot.contour(0) # This is going to plot the greenhouse. The 0 denotes the 0th indexed zone.

    plot.show_isosurfaces = True # Turn on isosurfaces
    cont.colormap_name = 'Magma' # Change colormap
    cont.legend.show = False # Turn legend on and off
    
    # You need these two lines below for contours so it can know to plot boundary faces or use "ALL" to plot every face. Again I use 0 to reference zone 0
    srf = plot.fieldmap(0).surfaces
    srf.surfaces_to_plot = SurfacesToPlot.BoundaryFaces
    
    plot.show_contour = True # Turn on contour
    plot.show_shade = False # Turn off shade
    
    # This code does the color cutoff stuff for the contour
    cont.color_cutoff.include_min = False
    cont.color_cutoff.min = 0.0
    cont.color_cutoff.include_max = True
    cont.color_cutoff.max = 1
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous # You can use Banded or continous. Just a visual thing.

    # Plotting the isosurface. again I use 0 here for 0th zone.
    iso = plot.isosurface(0)
    iso.show = True
    iso.definition_contour_group = cont
    iso.isosurface_selection = IsoSurfaceSelection.OneSpecificValue
    iso.isosurface_values = 1.0

    # Setting contour translucency
    plot.use_translucency = True
    plot.fieldmap(green_house).effects.surface_translucency = 70

    # Setting isosurface translucency
    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 70
    
    # Adding in time dependent data

    for i in range(int(ts/scale)):
        print("Adding zone {} of {}".format(i+1, int(ts/scale)))
        # tp.active_frame().delete_text(text)
        # Made a zone for each time step and set of data
        numRay = posTot[i*scale][:,0].shape[0]
        # print(numRay)
        if numRay > 0:
            shape = (numRay,1,1)
            rays = ds.add_ordered_zone('Light Rays',shape)
            rays.values('x')[:] = posTot[i*scale][:,0]
            rays.values('y')[:] = posTot[i*scale][:,1]
            rays.values('z')[:] = posTot[i*scale][:,2]
            rays.values('u')[:] = velTot[i*scale][:,0]/c
            rays.values('v')[:] = velTot[i*scale][:,1]/c
            rays.values('w')[:] = velTot[i*scale][:,2]/c
            rays.strand = 1 # Make sure the strand is the same for each set of transient data
            rays.solution_time = i # Set the time for which the data is associated with
        else:
            # print(numRay)
            shape = (1,1,1)
            rays = ds.add_ordered_zone('Light Rays',shape)
            rays.values('x')[:] = np.zeros((1,1,1))
            rays.values('y')[:] = np.zeros((1,1,1))
            rays.values('z')[:] = np.zeros((1,1,1))
            rays.values('u')[:] = np.zeros((1,1,1))
            rays.values('v')[:] = np.zeros((1,1,1))
            rays.values('w')[:] = np.zeros((1,1,1))
        
        # Same stuff below just have some if statements in there because Tecplot doesn't like zones with empty data sets.
        numSRay = sTot[i*scale][:,0].shape[0]
        
        if numSRay == 0:
            numSRay = 1
            shape = (1,1,1)
            solar = ds.add_ordered_zone('GHRays',shape)
            solar.values('x')[:] = np.zeros((1,1,1))
            solar.values('y')[:] = np.zeros((1,1,1))
            solar.values('z')[:] = np.zeros((1,1,1))
            solar.strand = 2
            solar.solution_time = i
        else:
            shape = (numSRay,1,1)
            solar = ds.add_ordered_zone('GHRays',shape)
            solar.values('x')[:] = sTot[i*scale][:,0]
            solar.values('y')[:] = sTot[i*scale][:,1]
            solar.values('z')[:] = sTot[i*scale][:,2]
            solar.strand = 2
            solar.solution_time = i
            
        numGRay = gTot[i*scale][:,0].shape[0]
        
        if numGRay == 0:
            numGRay = 1
            shape = (1,1,1)
            ground = ds.add_ordered_zone('GHRays',shape)
            ground.values('x')[:] = np.zeros((1,1,1))
            ground.values('y')[:] = np.zeros((1,1,1))
            ground.values('z')[:] = np.zeros((1,1,1))
            ground.strand = 3
            ground.solution_time = i
        else:
            shape = (numGRay,1,1)
            ground = ds.add_ordered_zone('GHRays',shape)
            ground.values('x')[:] = gTot[i*scale][:,0]
            ground.values('y')[:] = gTot[i*scale][:,1]
            ground.values('z')[:] = gTot[i*scale][:,2]
            ground.strand = 3
            ground.solution_time = i
            
        # tp.active_frame().add_latex(r'''July 1st, 2021, ''' + str(int(tsTot[i*scale])) + r''':00''' , (50,80), size=32, anchor=TextAnchor.Center)
        text = tp.active_frame().add_text('''July 1st, 2021, ''' + str(int(tsTot[i*scale])) + ''':00''' , (50,80), size=32, anchor=TextAnchor.Center,zone = solar)
        # text.solution_time = i
    # Applying scatter plot here. Notice I'm using 2 since this is the 3rd data group I added    
    scatter = plot.fieldmaps(2).scatter
    scatter.symbol_type = SymbolType.Geometry
    scatter.symbol().shape = GeomShape.Sphere
    scatter.color = Color.Red
    scatter.size = 1
    
    # Applying scatter plot here. Notice I'm using 3 since this is the 4th data group I added    
    scatter1 = plot.fieldmaps(3).scatter
    scatter1.symbol_type = SymbolType.Geometry
    scatter1.symbol().shape = GeomShape.Sphere
    scatter1.color = Color.Green
    scatter1.size = 1
    
    # Adding variables to vectors based on the velocity of the light vectors
    plot.vector.u_variable = ds.variable('u')
    plot.vector.v_variable = ds.variable('v')
    plot.vector.w_variable = ds.variable('w')
    plot.contour(0).variable = ds.variable('Greenhouse') # Adding color to the contour
    
    # Setting vector length
    plot.vector.use_relative = False
    plot.vector.length = 2
    
    # Turning vecotrs and scatter on
    plot.show_vector = True
    plot.show_scatter = True
        
    # Use the += lines to group together all of the zones with the same strand together 
    plot.active_fieldmaps += [1]
    plot.active_fieldmaps += [2]
    plot.active_fieldmap_indices = [0, 1, 2]
    plot.fieldmaps(2, 3).scatter.show = True # Only turn on scatter for things you want otherwise the contour will be scattered and the light rays and it looks ugly
    plot.fieldmaps(0, 1).scatter.show = False
    plot.active_fieldmaps += [3]

    if is_batch:
        tp.data.save_tecplot_plt('lightrays.plt', dataset=ds)
        tp.active_frame().save_stylesheet('lightrays.sty')
        tp.save_layout('lightrays.lay')
    # uncomment this to explort to video
    # tp.export.save_time_animation_mpeg4('Tecout.mp4', width=800, animation_speed=20)
print("Time to write Light Rays: ", time.time() - now)

# %% Plot green house

R1 = 0.5 # Greenhouse radius along e1 direction (m)
R2 = 5 # Greenhouse radius along e2 direction (m)
R3 = 0.5 # Greenhouse radius along e3 direction (m)

# Plotting grid parameters 
N = 40
L = 1

x = np.linspace(-L,L,N) # Define grid points in one dimension
z = np.linspace(0,L,N)

X,Y,Z = np.meshgrid(x,x,z)

if not is_batch:
    tp.session.connect()

shape = (N,N,N)

now = time.time()
with tp.session.suspend():
    # Create a new frame for the second dataset, That way you don't lose the original data
    # that took so long to import
    new_frame = tp.active_page().add_frame()
    new_frame.activate()
    ds = tp.active_frame().create_dataset('Green House Contour', ['x','y','z','GreenHouse','xr','yr','zr','u','v','w'])

    green_house = ds.add_ordered_zone('Green House',shape)

    green_house.values('x')[:] = X.ravel()
    green_house.values('y')[:] = Y.ravel()
    green_house.values('z')[:] = Z.ravel()
    # green_house.values('GreenHouse')[:] = (((np.abs(X.ravel()))/R1)**bestLam[2] + ((np.abs(Y.ravel()))/R2)**bestLam[3] + ((np.abs(Z.ravel()))/R3)**bestLam[4]) + np.abs(bestLam[5]*np.sin(bestLam[7]*np.abs(X.ravel()))) + np.abs(bestLam[6]*np.sin(bestLam[8]*np.abs(Y.ravel()))) + np.abs(bestLam[6]*np.sin(bestLam[8]*np.abs(Z.ravel())))
    green_house.values('GreenHouse')[:] = (((np.abs(X.ravel()))/R1)**bestLam[2] + ((np.abs(Y.ravel()))/R2)**bestLam[3] + ((np.abs(Z.ravel()))/R3)**bestLam[4]) + bestLam[5]*np.sin(bestLam[7]*X.ravel()) + bestLam[6]*np.sin(bestLam[8]*Y.ravel())
    
    # mag = np.linalg.norm(np.array([(X.ravel()*bestLam[2]*(np.abs(X.ravel())**(bestLam[2]-2)))/(np.abs(R1)**bestLam[2]) + (bestLam[5]*X.ravel()*np.sin(bestLam[5]*np.abs(X.ravel()))*np.cos(bestLam[5]*np.abs(X.ravel()))*np.abs(bestLam[7]))/(np.abs(X.ravel()*np.sin(bestLam[5]*np.abs(X.ravel())))),
    #                                (Y.ravel()*bestLam[3]*(np.abs(Y.ravel())**(bestLam[3]-2)))/(np.abs(R2)**bestLam[3]) + (bestLam[6]*Y.ravel()*np.sin(bestLam[6]*np.abs(Y.ravel()))*np.cos(bestLam[6]*np.abs(Y.ravel()))*np.abs(bestLam[8]))/(np.abs(Y.ravel()*np.sin(bestLam[6]*np.abs(Y.ravel())))),
    #                                ((Z.ravel())*bestLam[4]*(np.abs((Z.ravel()))**(bestLam[4]-2)))/(np.abs(R3)**bestLam[4])]),2,0)
    
    mag = np.linalg.norm(np.array([(X.ravel()*bestLam[2]*(np.abs(X.ravel())**(bestLam[2]-2)))/(np.abs(R1)**bestLam[2]) + bestLam[5]*bestLam[7]*np.cos(bestLam[7]*X.ravel()),
                                   (Y.ravel()*bestLam[3]*(np.abs(Y.ravel())**(bestLam[3]-2)))/(np.abs(R2)**bestLam[3]) + bestLam[6]*bestLam[8]*np.cos(bestLam[8]*X.ravel()),
                                   ((Z.ravel())*bestLam[4]*(np.abs((Z.ravel()))**(bestLam[4]-2)))/(np.abs(R3)**bestLam[4])]),2,0)
    
    # green_house.values('u')[:] = ((X.ravel()*bestLam[2]*(np.abs(X.ravel())**(bestLam[2]-2)))/(np.abs(R1)**bestLam[2]) + (bestLam[5]*X.ravel()*np.sin(bestLam[5]*np.abs(X.ravel()))*np.cos(bestLam[5]*np.abs(X.ravel()))*np.abs(bestLam[7]))/(np.abs(X.ravel())*np.abs(np.sin(bestLam[5]*np.abs(X.ravel())))))/mag
    # green_house.values('v')[:] = ((Y.ravel()*bestLam[3]*(np.abs(Y.ravel())**(bestLam[3]-2)))/(np.abs(R2)**bestLam[3]) + (bestLam[6]*Y.ravel()*np.sin(bestLam[6]*np.abs(Y.ravel()))*np.cos(bestLam[6]*np.abs(Y.ravel()))*np.abs(bestLam[8]))/(np.abs(Y.ravel())*np.abs(np.sin(bestLam[6]*np.abs(Y.ravel()))))/mag)
    # green_house.values('w')[:] = (((Z.ravel())*bestLam[4]*(np.abs((Z.ravel()))**(bestLam[4]-2)))/(np.abs(R3)**bestLam[4]))/mag
    
    green_house.values('u')[:] = -((X.ravel()*bestLam[2]*(np.abs(X.ravel())**(bestLam[2]-2)))/(np.abs(R1)**bestLam[2]) + bestLam[5]*bestLam[7]*np.cos(bestLam[7]*X.ravel()))/mag
    green_house.values('v')[:] = -((Y.ravel()*bestLam[3]*(np.abs(Y.ravel())**(bestLam[3]-2)))/(np.abs(R2)**bestLam[3]) + bestLam[6]*bestLam[8]*np.cos(bestLam[8]*Y.ravel()))/mag
    green_house.values('w')[:] = -(((Z.ravel())*bestLam[4]*(np.abs((Z.ravel()))**(bestLam[4]-2)))/(np.abs(R3)**bestLam[4]))/mag

    # log.info('setting plot type to Cart3D')
    tp.active_frame().plot_type = tp.constant.PlotType.Cartesian3D

    # plot = tp.active_frame().plot()

    # plot.show_shade = False
    # plot.show_contour = True
    # plot.show_isosurfaces = True

    plot = tp.active_frame().plot()
    cont = plot.contour(0)

    plot.vector.u_variable = ds.variable('u')
    plot.vector.v_variable = ds.variable('v')
    plot.vector.w_variable = ds.variable('w')
    
    # Setting vector length
    plot.vector.use_relative = False
    plot.vector.length = 5
    
    plot.show_isosurfaces = True
    cont.colormap_name = 'Magma'
    cont.legend.show = False
    srf = plot.fieldmap(0).surfaces
    srf.surfaces_to_plot = SurfacesToPlot.BoundaryFaces
    plot.show_contour = True
    plot.show_shade = False
    cont.plot
    cont.color_cutoff.include_min = False
    cont.color_cutoff.min = 0.0
    cont.color_cutoff.include_max = True
    cont.color_cutoff.max = 1
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous

    iso = plot.isosurface(0)
    iso.show = True
    iso.definition_contour_group = cont
    iso.isosurface_selection = IsoSurfaceSelection.OneSpecificValue
    iso.isosurface_values = 1.0

    plot.use_translucency = True
    plot.fieldmap(green_house).effects.surface_translucency = 70
    plot.show_vector = True

    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 70

    tp.active_frame().background_color = 7
    if is_batch:
        tp.data.save_tecplot_plt('greenhousecontour.plt', dataset=ds)
        tp.active_frame().save_stylesheet('greenhousecontour.sty')
        tp.save_layout('result.lay')
print("Time to write Green House Contour: ", time.time() - now)


#%% Plotting irradiance over single day

Pday = list()
time = list()
for hourtime in range(24):
    for minutetime in range(60):
        
        date = datetime.datetime(year, month, day, hour = hourtime, minute = minutetime, second=0,microsecond=0, tzinfo=timezone)
        altDeg = ps.get_altitude(latitude, longitude, date)
        aziDeg = ps.get_azimuth(latitude, longitude, date)
        altRad = altDeg * np.pi / 180 # Altitude angle of incoming rays (rad)
        aziRad = aziDeg * np.pi / 180 # Azimuth angle of incoming rays (rad)
        
        if altRad > 0 and altRad < np.pi: #if altitude angle is greater than 0 (sun is above horizon)
            time.append(hourtime+(minutetime/60))
            Pday.append(ps.radiation.get_radiation_direct(date,altDeg)) # Solar ray irradiance for given time and location (W/m^2)
            
Phour = list()
timehour = list()
for hourtime in range(24):
        
    date = datetime.datetime(year, month, day, hour = hourtime, minute = 0, second=0,microsecond=0, tzinfo=timezone)
    altDeg = ps.get_altitude(latitude, longitude, date)
    aziDeg = ps.get_azimuth(latitude, longitude, date)
    altRad = altDeg * np.pi / 180 # Altitude angle of incoming rays (rad)
    aziRad = aziDeg * np.pi / 180 # Azimuth angle of incoming rays (rad)
    
    if altRad > 0 and altRad < np.pi: #if altitude angle is greater than 0 (sun is above horizon)
        timehour.append(hourtime)
        Phour.append(ps.radiation.get_radiation_direct(date,altDeg)) # Solar ray irradiance for given time and location (W/m^2)
            
#%% Create irradiance plot

plt.figure(figsize = (15,7))
plt.plot(time,Pday)
plt.plot(timehour,Phour,'.',ms = 20)
plt.title("Solar irradiance on July 1, 2021 in Berkeley, CA")
plt.xlabel('Time (24 Hour Format)')
plt.xticks(np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
plt.ylabel(r'Solar irradiance $(\frac{W}{m^2})$')
plt.show()