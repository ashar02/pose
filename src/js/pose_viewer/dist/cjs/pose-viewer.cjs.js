'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

const index = require('./index-aba873a6.js');

/*
 Stencil Client Patch Browser v4.5.0 | MIT Licensed | https://stenciljs.com
 */
const patchBrowser = () => {
    const importMeta = (typeof document === 'undefined' ? new (require('u' + 'rl').URL)('file:' + __filename).href : (document.currentScript && document.currentScript.src || new URL('pose-viewer.cjs.js', document.baseURI).href));
    const opts = {};
    if (importMeta !== '') {
        opts.resourcesUrl = new URL('.', importMeta).href;
    }
    return index.promiseResolve(opts);
};

patchBrowser().then(options => {
  return index.bootstrapLazy([["pose-viewer.cjs",[[1,"pose-viewer",{"src":[1],"svg":[4],"width":[1],"height":[1],"aspectRatio":[2,"aspect-ratio"],"padding":[1],"thickness":[2],"background":[1],"loop":[1028],"autoplay":[4],"playbackRate":[1026,"playback-rate"],"currentTime":[1026,"current-time"],"duration":[1026],"ended":[1028],"paused":[1028],"readyState":[1026,"ready-state"],"error":[32],"syncMedia":[64],"getPose":[64],"nextFrame":[64],"play":[64],"pause":[64]},null,{"src":["srcChange"]}]]]], options);
});

exports.setNonce = index.setNonce;

//# sourceMappingURL=pose-viewer.cjs.js.map