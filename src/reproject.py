import os

# Chaos Co; E6ESDRKLIN01

os.system("echo Reprojecting TIFF file for Chaos Co")

os.system("gdalwarp -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
        -s_srs 'PROJCS['Equirectangular EUROPA',\
                        GEOGCS['GCS_EUROPA',\
                                DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],\
                                PRIMEM['Reference_Meridian',0],\
                                UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]\
                                ],\
                        PROJECTION['Equirectangular'],\
                        PARAMETER['standard_parallel_1',0],\
                        PARAMETER['central_meridian',180],\
                        PARAMETER['false_easting',0],\
                        PARAMETER['false_northing',0],\
                        UNIT['metre',1,AUTHORITY['EPSG','9001']],\
                        AXIS['Easting',EAST],\
                        AXIS['Northing',NORTH]\
                ]' \
        -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
        -multi -of GTiff \
        ../data/Chaos_Co/image/E6ESDRKLIN01_GalileoSSI_E_small.tif \
        ../data/Chaos_Co/image/E6ESDRKLIN01_GalileoSSI_E_reproj.tif"
        )

# Chaos ee, hh, ii, jj, kk; c17ESNERTRM01

chaos = ['ee', 'hh', 'ii', 'jj', 'kk']
os.system("echo Reprojecting TIFF files for Chaos ee and hh-kk")

for c in chaos:
        os.system(f"gdalwarp -co COMPRESS=DEFLATE -co TILED=YES \
                -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
                -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
                -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
                -multi -of GTiff -ot Float32 \
                ../data/Chaos_{c}/image/c17ESNERTRM01_GalileoSSI_E_small.tif \
                ../data/Chaos_{c}/image/c17ESNERTRM01_GalileoSSI_E_reproj.tif"
                )


# Chaos dd; c11ESREGMAP01

os.system("echo Reprojecting TIFF file  for Chaos dd")

os.system("gdalwarp -co COMPRESS=DEFLATE -co TILED=YES \
        -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
        -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
        -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
        -multi -of GTiff -ot Float32 \
        ../data/Chaos_dd/image/c11ESREGMAP01_GalileoSSI_E_small.tif \
        ../data/Chaos_dd/image/c11ESREGMAP01_GalileoSSI_E_reproj.tif"
        )


# Chaos ff, gg; c17ESREGMAP01

chaos = ['ff', 'gg']
os.system("echo Reprojecting TIFF files for Chaos ff and gg")

for c in chaos:
        os.system(f"gdalwarp -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
                -to INSERT_CENTER_LONG=TRUE  --config CENTER_LONG 290 \
                -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
                -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
                -multi -of GTiff -te 127.6723 -71.348 237 19.8846 \
                ../data/Chaos_{c}/image/c17ESREGMAP01_GalileoSSI_E_small.tif \
                ../data/Chaos_{c}/image/c17ESREGMAP01_GalileoSSI_E_reproj.tif"
                )


# Chaos A, B, C, E; c15ESREGMAP02

chaos = ['A', 'B', 'C', 'D', 'E']
os.system("echo Reprojecting TIFF files for Chaos A-E")

for c in chaos:
        os.system(f"gdalwarp -co COMPRESS=DEFLATE -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
                -to INSERT_CENTER_LONG=TRUE  --config CENTER_LONG 237 \
                -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
                -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
                -multi -of GTiff -te 127.6723 -71.348 237 19.8846 \
                ../data/Chaos_{c}/image/c15ESREGMAP02_GalileoSSI_E_small.tif \
                ../data/Chaos_{c}/image/c15ESREGMAP02_GalileoSSI_E_reproj.tif"
                )


# Chaos F, G, H, I; c17ESREGMAP02

chaos = ['F', 'G', 'H', 'I']
os.system("echo Reprojecting TIFF files for Chaos F-I")

for c in chaos:
        os.system(f"gdalwarp -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
                -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
                -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
                -multi -of GTiff -ot Float32 \
                ../data/Chaos_{c}/image/c17ESREGMAP02_GalileoSSI_E_small.tif \
                ../data/Chaos_{c}/image/c17ESREGMAP02_GalileoSSI_E_reproj.tif"
                )

# Chaos aa, bb; c11ESREGMAP01_c17ESNERTRM01_c17ESREGMAP01
# IMPORTANT - note that Chaos bb and aa plate labels exist across 3
# Galileo SSI RegMaps; this uses a PNG of combined images: c11ESREGMAP01, c17ESNERTRM01, and c17ESREGMAP01

chaos = ['aa', 'bb']
os.system("echo Reprojecting TIFF files for Chaos aa and bb")

for c in chaos:
        os.system(f"gdalwarp -co COMPRESS=DEFLATE -co BIGTIFF=IF_NEEDED -co NUM_THREADS=ALL_CPUS \
                -to INSERT_CENTER_LONG=TRUE  --config CENTER_LONG 180 \
                -s_srs 'PROJCS['Equirectangular EUROPA',GEOGCS['GCS_EUROPA',DATUM['D_EUROPA',SPHEROID['EUROPA_localRadius',1560800,0]],PRIMEM['Reference_Meridian',0],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']]],PROJECTION['Equirectangular'],PARAMETER['standard_parallel_1',0],PARAMETER['central_meridian',180],PARAMETER['false_easting',0],PARAMETER['false_northing',0],UNIT['metre',1,AUTHORITY['EPSG','9001']],AXIS['Easting',EAST],AXIS['Northing',NORTH]]' \
                -t_srs 'GEOGCS['GCS_Europa_2000',DATUM['D_Europa_2000',SPHEROID['Europa_2000_IAU_IAG',1562090,0,AUTHORITY['ESRI','107915']],AUTHORITY['ESRI','106915']],PRIMEM['Reference_Meridian',0,AUTHORITY['ESRI','108900']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['ESRI','104915']]' \
                -multi -of GTiff -te 110 -72 200 21 \
                ../data/Chaos_{c}/image/c11ESREGMAP01_c17ESNERTRM01_c17ESREGMAP01_GalileoSSI_E.tif \
                ../data/Chaos_{c}/image/c11ESREGMAP01_c17ESNERTRM01_c17ESREGMAP01_GalileoSSI_E_reproj.tif"
                )