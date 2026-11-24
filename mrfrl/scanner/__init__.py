"""The virtual scanner as a service.

``core``     server side: one episode = one ScannerSession (world, noise stream, budget, hash-chained log), replay
``service``  server side: HTTP front end, one process per episode
``client``   agent side: a single file with no import from this package (HTTP client + command line)
"""
