/// <reference types="node" />
import { PoseBodyModel, PoseHeaderModel } from "./types";
import { Buffer } from "buffer";
export * from './types';
export declare class Pose {
    header: PoseHeaderModel;
    body: PoseBodyModel;
    constructor(header: PoseHeaderModel, body: PoseBodyModel);
    static from(buffer: Buffer): Pose;
    static fromLocal(path: string): Promise<Pose>;
    static fromRemote(url: string, abortController?: AbortController): Promise<Pose>;
}
