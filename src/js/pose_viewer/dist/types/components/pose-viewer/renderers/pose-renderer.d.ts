import { PoseViewer } from "../pose-viewer";
import { PoseBodyFrameModel, PoseLimb, PosePointModel, RGBColor } from "pose-format";
export declare abstract class PoseRenderer {
  protected viewer: PoseViewer;
  constructor(viewer: PoseViewer);
  x(v: number): number;
  y(v: number): number;
  isJointValid(joint: PosePointModel): boolean;
  abstract renderJoint(i: number, joint: PosePointModel, color: RGBColor): any;
  renderJoints(joints: PosePointModel[], colors: RGBColor[]): any[];
  abstract renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor): any;
  renderLimbs(limbs: PoseLimb[], joints: PosePointModel[], colors: RGBColor[]): any[];
  renderFrame(frame: PoseBodyFrameModel): any;
  abstract render(frame: PoseBodyFrameModel): any;
}
