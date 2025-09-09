Northlight Favicon Pack
================================

Files
-----
- favicon.ico (16, 32, 48, 64 px)
- favicon-16.png
- favicon-32.png
- apple-touch-icon.png (180x180)
- android-chrome-192x192.png
- android-chrome-512x512.png
- site.webmanifest

How to use
----------
1) Drop all files into your site's public root (or the path you reference).
2) Add the following lines inside your HTML <head>:

<!-- Place in <head> -->
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">

Notes
-----
- The ICO contains multiple sizes for best cross-browser support.
- PNG variants cover Apple touch icon and PWA/Android launch icons.
