import { NextResponse } from "next/server";

export async function POST() {
  const isProd = process.env.NODE_ENV === "production";
  const res = NextResponse.json({ ok: true });
  const common = { httpOnly: true, secure: isProd, sameSite: "lax" as const, path: "/", maxAge: 0 };
  res.cookies.set({ name: "sp_access_token", value: "", ...common });
  res.cookies.set({ name: "sp_refresh_token", value: "", ...common });
  return res;
}