import { PoseBodyFrameModel, PosePointModel, RGBColor } from "pose-format";
import { PoseRenderer } from "./pose-renderer";
export declare class CanvasPoseRenderer extends PoseRenderer {
  ctx: CanvasRenderingContext2D;
  thickness: number;
  renderJoint(_: number, joint: PosePointModel, color: RGBColor): void;
  renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor): void;
  render(frame: PoseBodyFrameModel): any;
}
