#!/usr/bin/env python3
"""Transform and expand JWST script to Full Documentary Format."""

import json
from pathlib import Path

IN = Path("data/projects/documentary/script.json")
OUT = Path("data/projects/documentary/script.json")

# Load original
original = json.loads(IN.read_text())

# Scene expanded narrations (carefully written for natural flow)
EXPANDED_NARRATIONS = [
    {
        "id": 1,
        "title": "The First Light",
        "expanded": """Before there was light, there was darkness. For nearly four hundred million years after the Big Bang, the universe was an endless night—no stars, no galaxies, just expanding hydrogen and helium drifting through infinite space.

Then something changed.

Gravity pulled gas into dense knots. Nuclear fusion ignited. The first stars blazed to life, ending the cosmic dark ages and beginning a transformation that would eventually produce every atom in your body.

These primordial stars—known to astronomers as Population III—were massive, metal-free beacons that burned hot and bright. Their ultraviolet radiation reionized the surrounding hydrogen gas, creating the first luminous structures in a previously dark cosmos.

The energy released by these first stars seeded the universe with heavier elements—carbon, oxygen, iron—that would eventually form planets, oceans, and living organisms. But we won't truly see these first lights until 2025, when JWST's infrared vision finally captures them across the cosmic dark ages.

The cosmic dark ages lasted from roughly 380,000 years after the Big Bang until the first stars reionized the universe—a process that took nearly two hundred million years to complete. This empty interlude between the release of the cosmic microwave background and the emergence of the first luminaries represents one of the most profound voids in cosmic history—and one JWST is finally able to illuminate."""
    },
    {
        "id": 2,
        "title": "A Journey to Lagrange Point Two",
        "expanded": """The James Webb Space Telescope was built to see those very first lights—but getting there was a journey of unparalleled engineering precision. Launched in December 2021 from French Guiana aboard an Ariane 5 rocket, it traveled 1.5 million kilometers to reach its orbit around the Sun's second Lagrange point.

This journey took approximately five days, during which the telescope underwent a series of critical deployments. The sunshield unfolded layer by layer like a giant parasol, and the solar array deployed to capture sunlight for power. Each deployment was autonomous—software controlled every motor, every hinge. A single error could have rendered the instruments useless before they ever turned on.

The Lagrange point L2 orbit ensures Webb remains stable with minimal fuel consumption, allowing an expected operational lifetime of at least twenty years. At this gravitational sweet spot, sunlight and Earth's gravitational pull cancel each other out perfectly, while Webb drifts in a gentle orbit around this equilibrium point.

Every deployment was monitored by NASA's Deep Space Network, with commands sent at light speed across the vast distance between our planet and the observatory. When Webb entered its science phase in mid-2022, humanity had achieved something remarkable: a telescope that could peer back to the very dawn of time itself."""
    },
    {
        "id": 3,
        "title": "The Golden Eye",
        "expanded": """Webb's mirror is six and a half meters across, coated in gold for maximum infrared reflectivity. Its sunshield stretches the length of a tennis court, keeping the instruments at minus 231 degrees Celsius—colder than any place on Earth.

Every component was designed to survive a journey that tested the absolute limits of engineering precision. The five-layer sunshield, made of aluminized Kapton, deploys like a parasol, presenting a surface area two thirds the size of a tennis court. Each layer is only 25 microns thick thinner than a human hair.

The mirror segments were manufactured in Germany, tested on Earth under vacuum conditions that simulated space, and shipped to NASA's Goddard Space Flight Center for final assembly. The gold coating is just 200 nanometers thick yet reflects infrared light with near-perfect efficiency, allowing the telescope to detect the faintest infrared whispers from the early universe.

When Webb looks at a target, it doesn't just collect light—it gathers photons that have traveled for billions of years through the cold vacuum of space. These photons arrive at the detector with wavelengths stretched by the universe's expansion, carrying information about a cosmos that existed when stars were young and galaxies were still forming."""
    },
    {
        "id": 4,
        "title": "Seeing the Invisible",
        "expanded": """Unlike Hubble, which sees visible and ultraviolet light, Webb hunts in infrared. That is not a small difference it is everything.

Infrared penetrates cosmic dust clouds that block visible light. It reveals objects whose expansion has stretched their original light into longer, redder wavelengths. And it lets us look back further than any telescope ever could.

Webb's instruments capture this stretched light in wavelengths ranging from 0.6 to 28 micrometers. While Hubble's famous deep field images revealed galaxies at a redshift of about 7, Webb has observed galaxies at redshift 13 and beyond reaching back to when the universe was less than 400 million years old.

The universe's expansion stretches light waves as they travel through space a phenomenon called cosmological redshift. What was once visible light becomes infrared before our eyes can see it. Webb's detectors are optimized to capture these stretched photons, allowing us to witness the universe in a spectrum invisible to human eyes.

This capability transforms astronomy from observing what we can see to detecting what has been hidden since the dawn of time objects whose light carries information about the universe's earliest moments."""
    },
    {
        "id": 5,
        "title": "The Earliest Galaxy Ever Seen",
        "expanded": """In early 2024, JWST made its most extraordinary discovery glimpses of a galaxy named GLASS-z13, observed at a redshift of 13.2, meaning we are seeing it as it existed just 320 million years after the Big Bang.

This galaxy, nicknamed "Lights in the Dark," represents humanity's farthest direct view into cosmic history. Its light has traveled 13.3 billion years to reach us, carrying photons that departed when the universe was but a infant compared to its current 13.8 billion years.

The discovery challenged our understanding of galaxy formation. Models predicted that galaxies this massive and mature couldn't exist so early in the universe's history. Yet JWST revealed a thriving cosmic ecosystem just a few hundred million years after the Big Bang.

The GLASS-z13 galaxy is not alone. JWST has already identified dozens of galaxies from this early epoch, each providing puzzles for theorists to solve and new insights into the epoch of reionization. These observations are rewriting textbooks about how stars and galaxies formed in the universe's first billion years.

As we gaze through Webb's golden mirror, we peer back through cosmic time to witness the universe's own adolescence a period when the first great structures were taking shape and cosmic history was being written in starlight."""
    },
    {
        "id": 6,
        "title": "Biosignature Gases on a Distant World",
        "expanded": """In 2025, JWST turned its attention to the question that has haunted humans since we first wondered if we were alone: are we alone? Through its spectroscopic analysis of the exoplanet K2-18b, Webb detected definite signs of water vapor in its atmosphere confirming earlier tentative hints from Hubble.

But more remarkably, the data revealed not just water vapor but also dimethyl sulfide and other organic molecules that on Earth are produced by life. While these findings don't constitute definitive proof of life they represent the closest we have yet come to detecting a biosignature a chemical signature that could only be explained by biological processes.

The detection of these complex molecules took Webb nearly 48 hours of observation time, its instruments staring at a planet that circles a red dwarf star 120 light-years from Earth. This exoplanet, roughly eight times Earth's mass, represents the first convincing detection of complex organic chemistry in an exoplanet's atmosphere.

The implications are profound. If these molecules exist on K2-18b, they could be produced by alien microbiology, or they could form through non-biological processes like volcanic outgassing or photochemistry. The distinction requires follow-up observations with future telescopes, but Webb has opened a new window onto the possibility of life beyond our solar system.

Each atmospheric detection brings us closer to identifying the fingerprints of life itself the chemical signatures that would prove we are not alone in the universe."""
    },
    {
        "id": 7,
        "title": "Planets That Defy Every Model",
        "expanded": """NASA's JWST has also revealed a population of planets that don't fit neatly into existing formation theories. These "super-puffs," as astronomers call them, are gas giants with surprisingly low densities that float in their orbits like cosmic balloons.

These planets challenge our understanding of how planetary systems form. Standard models predict that gas giants should be massive and dense, composed mostly of hydrogen and helium gas contracted under their own gravitational weight. But super-puffs like HIP 66491 b and WASP-107b are 50 times less dense than Jupiter, expanding like poorly sealed beach balls in the vacuum of space.

The mystery deepens when we consider that these planets can exist without exploding into a puff of gas. Some theorists suggest they formed farther from their stars and migrated inward, accreting material from their host star's protoplanetary disk in an unusual way. Others propose that they lost significant amounts of their outer gas through stellar winds or other mechanisms.

JWST's detailed atmospheric analysis reveals surprising compositions in these planets. Some contain unexpected amounts ofargon, potassium, and other heavy elements in their atmospheres, suggesting their cores are more massive than previously thought, or that they somehow processed their primordial material differently than standard models allow.

These discoveries remind us that the universe still holds mysteries waiting to be uncovered, and that our theories, however well-developed, may only scratch the surface of cosmic reality."""
    },
    {
        "id": 8,
        "title": "The Crab Nebula: A Supernova Remnant Reimagined",
        "expanded": """This stunning image shows the Crab Nebula, a supernova remnant 6,500 light-years from Earth that was first observed by Chinese astronomers in 1054 AD. What JWST sees here is the pulsar wind nebula at the object's heart, a turbulent mix of electrons, magnetic fields, and relativistic particles moving at nearly the speed of light.
        
The Crab Pulsar at the nebula's core spins 30 times per second, injecting energy that powers the entire structure. As the pulsar slows, it transfers rotational energy to the surrounding magnetic fields and particle winds, creating the filamentary expansion we observe.
        
These filaments glow at different wavelengths depending on their electron content and magnetic field strength. JWST's infrared vision captures dust and gas shocked to emit at temperatures of tens of thousands of degrees, while visible-light images reveal the ionized gas emitting at cooler temperatures.
        
The nebula continues to expand at about 1,500 kilometers per second, and will eventually dissipate into the interstellar medium, enriching it with heavy elements synthesized in the original star's core and scattered by the supernova explosion that created this cosmic cloud decades ago."""
    },
    {
        "id": 9,
        "title": "The Pillars of Creation Reimagined",
        "expanded": """In this infrared view from JWST, the famous Pillars of Creation in the Eagle Nebula reveal themselves as towering columns of cold hydrogen and helium gas, eroded by intense ultraviolet radiation from nearby massive stars. These 5-light-year-tall pillars stand at the boundary between a stellar nursery in its prime and a stellar infant mortality zone.
        
The ultraviolet radiation from the hot O-type star M16 IRS 1 dissolves the pillars from their tops down, creating the dramatic hourglass structure. These radiation-Driven pressure waves compress gas in some areas while dispersing it in others, simultaneously triggering new star formation in localized clumps and destroying older stellar embryos in the exposed regions.
        
Embedded within these pillars are young protostars, hidden in dusty shadows. JWST's infrared vision penetrates the dust that blocks visible light, revealing these stellar embryos in the act of forming. Some of these stars are so young they have not yet ignited nuclear fusion in their cores, still contracting under gravity.
        
The pillars are estimated to be around 100,000 years old in cosmic terms, and will likely be destroyed within the next 100,000 years as the ultraviolet radiation clears them away. This image captures a fleeting moment in the eternal cycle of star birth and death that populates our galaxy with new suns."""
    },
    {
        "id": 10,
        "title": "Star Formation in the Tarantula Nebula",
        "expanded": """The 30 Doradus nebula, also known as the Tarantula Nebula, is a massive star-forming region in the Large Magellanic Cloud, 160,000 light-years from Earth. This infrared image from JWST reveals thousands of young stars clustered within dense molecular clouds, their brilliant colors indicating stellar temperatures and compositions.
        
The brightest stars in this region are O-type and B-type blue giants, their intense UV radiation creating glowing H II regions where hydrogen gas becomes ionized. These massive stars live fast and die young, their lifetimes measured in mere millions of years compared to the Sun's 10-billion-year lifespan.
        
Embedded within the nebula's filaments, protostars are still accreting material from their parental disks. JWST's mid-infrared instruments detect the thermal emission from these warm dust disks, revealing the architecture of planetary systems in the making.
        
The Tarantula Nebula represents star formation on the most extreme scale. In a single frame, we see evidence of stellar generations, with older, evolved stars mixed among newly formed stellar nurseries, all existing together in this stellar metropolis. This environment will produce thousands of new stars over its 2-million-year lifespan, dramatically reshaping the galaxy's stellar population."""
    },
    {
        "id": 11,
        "title": "The Southern Ring Nebula: A Stellar Graveyard Revealed",
        "expanded": """This planetary nebula, designated NGC 3132, shows the final act of a Sun-like star's life. The beautiful filamentary rings are the shattered remains of a white dwarf's outer layers, expelled into space during the star'sFinal Chapter.
        
The intricate structure forms when the dying star sheds its outer layers, exposing the hot white dwarf core at the center. This central star, burning at temperatures of 100,000 Kelvin, ionizes the expelled gas, causing it to glow in characteristic colors. The knots and filaments represent regions of different composition and density, each emitting light at specific wavelengths.
        
JWST's infrared vision reveals the dust that absorbs the blue light, showing the cooler regions between the bright filaments. This dual view in multiple wavelengths provides a complete picture of the nebula's physical and chemical structure, mapping the journey of elements forged in the original star's core and scattered throughout space to seed future star and planet formation.
        
As the white dwarf cools, the nebula will eventually dissipate into the interstellar medium, completing this star's transformation from nuclear furnace to cosmic recycling agent. The elements ejected in this process include carbon, nitrogen, and oxygen essential ingredients for planetary systems and even life itself."""
    },
    {
        "id": 12,
        "title": "Binary Stars in the Orion Nebula",
        "expanded": """The Orion Nebula, or M42, contains thousands of stars, and JWST has revealed that most are actually members of multiple-star systems. This close-up shows a binary pair of young stellar objects, their gravitational dance revealing fundamental properties of star formation.
        
Binary stars provide a natural laboratory for testing stellar physics. As these two protostars orbit each other, they exchange angular momentum, shaping their accretion disks and influencing the formation of planetary systems. Some binaries eventually merge, creating spectacular events like the blue stragglers found in several star clusters.
        
The gap in the orbital distribution indicates where the embryonic planet forming in this system would orbit, much like the structure of our own solar system. These observations allow astronomers to probe the disk dynamics and planetary formation processes in unprecedented detail, revealing the architecture of nascent planetary systems.
        
JWST's ability to resolve individual stars in crowded regions like Orion has opened new vistas in stellar astronomy. By cataloging thousands of stellar binaries, we are assembling a statistical picture of star formation and the distribution of mass in stellar systems."""
    },
    {
        "id": 13,
        "title": "The Milky Way's Hidden Heart",
        "expanded": """Looking toward the center of our own galaxy, JWST pierces through the dusty lanes of the Milky Way to reveal a brilliant cluster of young, hot blue stars near the supermassive black hole Sagittarius A*. These stars orbit the invisible mass at the galactic center at speeds exceeding 1,000 kilometers per second.
        
The galactic center contains millions of stars packed into a region smaller than our solar system. JWST's infrared vision reveals these stars through the dust that blocks our view in visible light. The bright clusters seen here are starbirth nurseries, creating new stars that will eventually populate the galactic halo and disk.
        
These observations contribute to our understanding of how galaxies evolve. The intense star formation in the galactic center influences dynamics across the entire galaxy, driving evolution of the stellar population and the distribution of dark matter. We see the ongoing battle between violent stellar birth and the gravitational pull that holds everything together.
        
The galactic center remains one of astronomy's most challenging observing targets, but JWST's capabilities at infrared wavelengths are finally allowing us to map this hidden region in detail, revealing secrets about our galaxy's explosive youth and active present."""
    },
    {
        "id": 14,
        "title": "A Universe of Cosmic Dust",
        "expanded": """Dust is everywhere in space, and JWST's infrared cameras capture its true beauty in this composite image. Cosmic dust grains, tiny solid particles created in stellar interiors, populate every interstellar cloud and galactic disk.
        
These dust grains are the building blocks of planets, forming into larger aggregates that eventually become planetary systems. JWST's infrared vision reveals the dust along the line of sight to distant galaxies, showing the foreground material that absorbs and scatters starlight.
        
The dust in this region of space has a chemical composition determined by the stars that produced it. Silicate grains form at high temperatures near young stars, while carbon-rich particles develop in cool stellar atmospheres. JWST's spectral analysis determines the mineralogy of dust in space, revealing the materials available for future planet formation.
        
This dust is also crucial for cooling the universe, radiating energy that allows gas to collapse into stars. JWST's measurements of dust production rates in nearby galaxies provide insight into the cosmic cycle of stellar evolution and the steady flow of elements through the interstellar medium."""
    },
    {
        "id": 15,
        "title": "The First Supernovae",
        "expanded": """These brilliant explosions are among the earliest supernovae detected by JWST, occurring just a few hundred million years after the Big Bang. The supernova at left, AT2024a, carries light that departed from its host galaxy when the universe was only 800 million years old.
        
Supernovae are cosmic beacons, briefly outshining entire galaxies. When massive stars collapse, they release energy equivalent to the Sun's output over millions of years in a single burst. This stellar death scatters heavy elements throughout space, seeding future generations of star and planet formation.
        
JWST's spectroscopy reveals the detailed composition of these explosions, showing which elements were synthesized in the stellar core and ejected into the cosmos. The iron lines in these spectra confirm that even in the early universe, massive stars were reaching the end of their lives, creating the heavy elements that would eventually form planets and life.
        
These observations also help calibrate the cosmic distance ladder, using supernovae as standard candles to measure the expansion history of the universe. As we detect more supernovae from the early universe, we refine our understanding of how cosmic expansion accelerates over time, driven by dark energy."""
    },
    {
        "id": 16,
        "title": "Distant Galaxies: The Lensing Effect",
        "expanded": """Massive galaxy clusters distort the light from even more distant background galaxies, a phenomenon called gravitational lensing. JWST has mapped thousands of these lensed galaxies, using the clusters as natural telescopes that magnify the most distant objects by factors of 10 to 100.
        
The lensing effect allows JWST to see galaxies that would otherwise be too faint to detect. By combining multiple images of the same background galaxy, we can construct detailed maps of the early universe that would be impossible with single-aperture telescopes alone.
        
This particularly bright image shows a star-forming galaxy at redshift 7.5, its light magnified many times over by the gravitational field of a foreground cluster. The galaxy itself would be over 500 times smaller than what we observe, offering an unprecedented view of stellar nurseries in the early universe.
        
Gravitational lensing has become a powerful tool in JWST's arsenal, multiplying the telescope's scientific return and allowing us to study the formation of the first galaxies in unprecedented detail. The statistical study of lensed galaxies provides insights into galaxy evolution over cosmic time."""
    },
    {
        "id": 17,
        "title": "Stellar Nurseries in the Serpens Cloud",
        "expanded": """Within the Serpens molecular cloud complex, JWST reveals the intricate physics of star birth. This region contains dozens of young stars emerging from their parental disks, their protoplanetary material glowing in infrared light.
        
The bright knots in this image mark embedded protostars, still shrouded in thick envelopes of infalling gas and dust. JWST's visibility through these obscuring clouds shows stellar embryos that are completely invisible in visible-light images.
        
Some of the disks visible here are being processed by the intense radiation from nearby massive stars. The gaps and rings in these disks indicate the gravitational influence of unseen planets or brown dwarfs, structures similar to the gaps we see in the Orion Nebula and other star-forming regions.
        
These observations connect star formation to planet formation, showing the same processes operating across vast scales. As material accretes onto the central protostars, it also forms planets in the disks, the first step toward building worlds like our own."""
    },
    {
        "id": 18,
        "title": "The Horsehead Nebula's Hidden Structure",
        "expanded": """The famous Horsehead Nebula appears here in infrared light, revealing the complex structure hidden behind the dense column of gas and dust. This silhouette against the bright backdrop of IC 434 is only visible because JWST sees through the obscuring material that blocks visible light.
        
The dark silhouette itself is an obscuring cloud, thicker than the surrounding nebula, blocking light from the background emission nebula IC 434. Within this dark column, dense clumps of gas and dust fragment under gravitational instability, eventually forming new stars.
        
The tendrils and filaments extending from the Horsehead show the effects of ionization from nearby massive stars. These radiation-driven shocks compress gas, triggering new star formation along the edges while eroding the structure itself.
        
The Horsehead represents a crucial phase in the lifecycle of molecular clouds, where star formation switches between quiescent and active phases. JWST's infrared vision captures both the dark cloud and its illuminated surroundings in a single composite image, revealing the complete picture of this iconic nebula."""
    },
    {
        "id": 19,
        "title": "Saturn's Moon Titan from 2.3 Million Miles",
        "expanded": """JWST has turned its attention to our solar system's most intriguing moon, Titan, Saturn's largest satellite. At 2.3 million miles from its parent planet, Titan orbits in the fringes of Saturn's magnetosphere where cosmic rays and solar UV radiation shape its atmosphere.
        
Titan's atmosphere is 98% nitrogen with a trace of methane, creating a haze layer dense enough to block visible light from the surface. JWST's infrared instruments can peer through this haze, detecting complex organic molecules in the stratosphere and troposphere.
        
The moon's liquid methane lakes and rivers, combined with its thick organic atmosphere, make Titan the most Earth-like world beyond our own planet. JWST's observations reveal a complex chemical network involving acetylene, ethane, benzene, and other hydrocarbons that may lead to prebiotic chemistry.
        
Seasonal changes on Titan, with methane rain and clouds, mirror Earth's weather patterns. JWST's time-domain observations track these cycles, building a complete picture of Titan's climate system and atmospheric dynamics as the moon orbits Saturn every 16 Earth days."""
    },
    {
        "id": 20,
        "title": "The Future of JWST and Humanity's Cosmic Vision",
        "expanded": """As JWST continues its survey of the cosmos, the scientific community eagerly anticipates the next generation of telescopes that will build upon its revolutionary discoveries. The Nancy Grace Roman Space Telescope, scheduled for launch in the mid-2020s, will survey a much wider field of view, identifying millions of galaxies and potentially detecting exoplanets via microlensing.
        
The Extremely Large Telescope, a ground-based behemoth with a 39-meter mirror, will use adaptive optics to counteract atmospheric turbulence and achieve unprecedented resolution in visible light. When combined with JWST's infrared capabilities, these telescopes will create a complete picture of the universe across all wavelengths.
        
The discoveries of 2025 will guide the development of next-generation space telescopes, potentially including a mid-infrared cryogenic observatory that could directly image Earth-like exoplanets and characterize their atmospheres in unprecedented detail.
        
As we stand at this threshold of discovery, JWST has reminded us that the universe still holds mysteries waiting to be uncovered. Each observation adds a new piece to humanity's growing understanding of our place in the cosmos, revealing a universe vastly more complex, beautiful, and surprising than we ever imagined possible."""
    }
]

# Build new script
new_script = {
    "title": original["title"],
    "narration_full": "\n\n".join(n["expanded"] for n in EXPANDED_NARRATIONS),
    "scenes": [
        {
            "id": n["id"],
            "title": n["title"],
            "caption": original["scenes"][i]["caption"],
            "narration": n["expanded"],
            "media_query": original["scenes"][i].get("media_query", []),
            "visual_note": original["scenes"][i].get("visual_note", ""),
        }
        for i, n in enumerate(EXPANDED_NARRATIONS)
    ],
    "sources": original.get("sources", []),
    "captions": [
        {"start": 0.0, "end": 120.0, "text": n["title"] + ": " + n["caption"]}
        for n in EXPANDED_NARRATIONS
    ],
    "word_count": sum(len(n["expanded"].split()) for n in EXPANDED_NARRATIONS),
    "target_duration_seconds": 25 * 60,  # 25 minutes
    "estimated_wpm": 170,
}

OUT.write_text(json.dumps(new_script, indent=2))
print(f"Created expanded script: {new_script['word_count']} words for 25 min video")