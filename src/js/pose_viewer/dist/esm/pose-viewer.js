import { p as promiseResolve, b as bootstrapLazy } from './index-b27121e8.js';
export { s as setNonce } from './index-b27121e8.js';

/*
 Stencil Client Patch Browser v4.5.0 | MIT Licensed | https://stenciljs.com
 */
const patchBrowser = () => {
    const importMeta = import.meta.url;
    const opts = {};
    if (importMeta !== '') {
        opts.resourcesUrl = new URL('.', importMeta).href;
    }
    return promiseResolve(opts);
};

patchBrowser().then(options => {
  return bootstrapLazy([["pose-viewer",[[1,"pose-viewer",{"src":[1],"svg":[4],"width":[1],"height":[1],"aspectRatio":[2,"aspect-ratio"],"padding":[1],"thickness":[2],"background":[1],"loop":[1028],"autoplay":[4],"playbackRate":[1026,"playback-rate"],"currentTime":[1026,"current-time"],"duration":[1026],"ended":[1028],"paused":[1028],"readyState":[1026,"ready-state"],"error":[32],"syncMedia":[64],"getPose":[64],"nextFrame":[64],"play":[64],"pause":[64]},null,{"src":["srcChange"]}]]]], options);
});

//# sourceMappingURL=pose-viewer.js.map